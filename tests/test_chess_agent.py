"""Unit tests for bots/chess_agent.py. Run with: pytest tests/test_chess_agent.py"""
import random
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import chess
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bots"))

import chess_agent  # noqa: E402

CHECK_POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1PN2/PPP2PPP/R1BQK2R w KQkq - 0 9",
    "8/5pk1/6p1/8/2Q5/6P1/5PK1/3q4 b - - 0 40",
]
WHITE_UP_FEN = "rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3"
BLACK_UP_FEN = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 3"
EN_PASSANT_PIN_FEN = "8/8/8/KPp4r/8/8/8/7k w - c6 0 2"
MATE_IN_TWO_FEN = "7k/8/8/8/8/8/R7/1R5K w - - 0 1"
HANGING_QUEEN_FEN = "r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1PN2/PPP2PPP/R1BQK2R w KQkq - 0 9"
PERFT_CASES = [
    (chess.STARTING_FEN, [20, 400, 8902]),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [48, 2039, 97862]),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379]),
]


def mailbox_from_fen(fen: str) -> chess_agent.MailboxBoard:
    """Return a Core B board loaded from a FEN."""
    mailbox = chess_agent.MailboxBoard()
    mailbox.load_board(chess.Board(fen))
    return mailbox


def perft(mailbox: chess_agent.MailboxBoard, depth: int) -> int:
    """Count leaf positions with Core B's pseudo-legal generator and legality test."""
    if depth == 0:
        return 1
    total = 0
    side = mailbox.side
    for move in mailbox.generate_moves(False):
        mailbox.make_move(move)
        if not mailbox.is_attacked(mailbox.king_squares[side], side ^ 1):
            total += perft(mailbox, depth - 1)
        mailbox.unmake_move()
    return total


def make_observation(board: chess.Board, time_remaining: float = 5.0) -> dict:
    """Build an observation like the judge's for a python-chess board."""
    legal_moves = []
    for move in board.legal_moves:
        legal_moves.append(move.uci())
    legal_moves.sort()
    history = []
    for move in board.move_stack:
        history.append(move.uci())
    return {
        "game": "chess",
        "game_number": 1,
        "games_total": 2,
        "color": "white" if board.turn == chess.WHITE else "black",
        "fen": board.fen(),
        "legal_moves": legal_moves,
        "history": history,
        "ply": board.ply(),
        "is_check": board.is_check(),
        "time_remaining_s": time_remaining,
        "opponent_time_remaining_s": time_remaining,
    }


@pytest.fixture(autouse=True)
def fresh_state() -> Iterator[None]:
    """Give every test a new game so module state never leaks between tests."""
    chess_agent.STATE.reset()
    yield
    chess_agent.STATE.reset()


def test_evaluate_ranks_material() -> None:
    balanced = chess_agent.evaluate(chess.STARTING_FEN)
    white_up = chess_agent.evaluate(WHITE_UP_FEN)
    black_up = chess_agent.evaluate(BLACK_UP_FEN)
    assert balanced == 0
    assert white_up > balanced > black_up
    assert isinstance(white_up, int) and 800 < white_up < 1200


def test_evaluate_scores_finished_games() -> None:
    assert chess_agent.evaluate("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1") == 100000
    assert chess_agent.evaluate("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1") == 0
    assert chess_agent.evaluate("8/8/8/4k3/8/8/8/4K3 w - - 0 1") == 0


@pytest.mark.parametrize("fen", CHECK_POSITIONS)
def test_alphabeta_matches_minimax(fen: str) -> None:
    plain = chess_agent.minimax(fen, 2)
    pruned = chess_agent.alphabeta(fen, 2)
    assert plain["value"] == pruned["value"]
    assert pruned["nodes"] <= plain["nodes"]


def test_alphabeta_prunes_and_deepens() -> None:
    fewer = False
    for fen in CHECK_POSITIONS:
        pruned_nodes = chess_agent.alphabeta(fen, 2)["nodes"]
        fewer = fewer or pruned_nodes < chess_agent.minimax(fen, 2)["nodes"]
    assert fewer
    shallow = chess_agent.alphabeta(CHECK_POSITIONS[1], 1)["nodes"]
    deep = chess_agent.alphabeta(CHECK_POSITIONS[1], 2)["nodes"]
    assert deep > shallow


def test_book_lines_are_legal() -> None:
    for line, weight in chess_agent.BOOK_LINES:
        board = chess.Board()
        for uci in line.split():
            move = chess.Move.from_uci(uci)
            assert board.is_legal(move), f"{uci} is illegal in {line}"
            board.push(move)
        assert weight > 0


def test_book_covers_main_replies() -> None:
    start = chess_agent.OPENING_BOOK[chess_agent.book_key(chess.Board())]
    assert sorted(start) == [["d2d4", 45], ["e2e4", 55]]
    for first_move in ("e2e4", "d2d4", "c2c4", "g1f3"):
        board = chess.Board()
        board.push_uci(first_move)
        assert chess_agent.book_key(board) in chess_agent.OPENING_BOOK


@pytest.mark.parametrize("fen, expected", PERFT_CASES)
def test_perft(fen: str, expected: list) -> None:
    mailbox = mailbox_from_fen(fen)
    for depth, count in enumerate(expected, 1):
        assert perft(mailbox, depth) == count
    assert mailbox.matches(chess.Board(fen))


def test_random_games_match_python_chess() -> None:
    generator = random.Random(2026)
    for game in range(30):
        board = chess.Board()
        mailbox = chess_agent.MailboxBoard()
        while not board.is_game_over() and board.ply() < 300:
            generated = set()
            for move in mailbox.legal_moves():
                generated.add(chess_agent.move_to_uci(move))
            expected = set()
            for move in board.legal_moves:
                expected.add(move.uci())
            assert generated == expected, board.fen()
            move = generator.choice(list(board.legal_moves))
            hash_before = mailbox.hash_value
            packed = mailbox.parse_uci(move.uci())
            mailbox.make_move(packed)
            board.push(move)
            assert mailbox.matches(board), board.fen()
            mailbox.unmake_move()
            board.pop()
            assert mailbox.matches(board) and mailbox.hash_value == hash_before
            mailbox.make_move(packed)
            board.push(move)


def test_uci_round_trip() -> None:
    for fen, counts in PERFT_CASES:
        mailbox = mailbox_from_fen(fen)
        for move in mailbox.legal_moves():
            assert mailbox.parse_uci(chess_agent.move_to_uci(move)) == move


def test_promotion_uci_names_the_piece() -> None:
    mailbox = mailbox_from_fen("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    names = set()
    for move in mailbox.legal_moves():
        names.add(chess_agent.move_to_uci(move))
    assert {"a7a8q", "a7a8r", "a7a8b", "a7a8n"} <= names


def test_castling_mask_clears_rights() -> None:
    mailbox = mailbox_from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    mailbox.make_uci_move("h1h8")
    assert mailbox.castling == chess_agent.CASTLE_WHITE_QUEEN | chess_agent.CASTLE_BLACK_QUEEN
    mailbox.unmake_move()
    mailbox.make_uci_move("e1d1")
    assert mailbox.castling == chess_agent.CASTLE_BLACK_KING | chess_agent.CASTLE_BLACK_QUEEN
    mailbox.unmake_move()
    assert mailbox.castling == 15


def test_castling_through_check_is_not_generated() -> None:
    mailbox = mailbox_from_fen("4k3/8/8/8/8/8/5r2/R3K2R w KQ - 0 1")
    names = set()
    for move in mailbox.legal_moves():
        names.add(chess_agent.move_to_uci(move))
    assert "e1g1" not in names
    assert "e1c1" in names


def test_en_passant_rank_pin() -> None:
    board = chess.Board(EN_PASSANT_PIN_FEN)
    mailbox = mailbox_from_fen(EN_PASSANT_PIN_FEN)
    names = set()
    for move in mailbox.legal_moves():
        names.add(chess_agent.move_to_uci(move))
    assert "b5c6" not in names
    expected = set()
    for move in board.legal_moves:
        expected.add(move.uci())
    assert names == expected


def test_repetition_is_detected() -> None:
    mailbox = chess_agent.MailboxBoard()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        assert not mailbox.is_repetition()
        mailbox.make_uci_move(uci)
    assert mailbox.is_repetition()


def test_zobrist_keys_ignore_global_seed() -> None:
    random.seed(1)
    first = chess_agent.build_zobrist_keys()
    random.seed(2)
    second = chess_agent.build_zobrist_keys()
    assert first == second
    assert first[3] == chess_agent.ZOBRIST_SIDE


def test_mop_up_prefers_cornered_king() -> None:
    centred = chess_agent.evaluate("8/8/8/3k4/8/8/8/4K2Q w - - 0 1")
    cornered = chess_agent.evaluate("k7/8/8/8/8/8/8/4K2Q w - - 0 1")
    assert cornered > centred
    mailbox = mailbox_from_fen("k7/8/8/8/8/8/8/4K2Q w - - 0 1")
    assert mailbox.white_score() == cornered


def test_core_b_static_score_matches_evaluate() -> None:
    for fen in CHECK_POSITIONS + [PERFT_CASES[1][0], PERFT_CASES[4][0]]:
        assert mailbox_from_fen(fen).white_score() == chess_agent.evaluate(fen)


def test_time_budget_never_exceeds_hard_fraction() -> None:
    for remaining in (0.6, 0.8, 1.0, 2.0, 5.0, 10.0):
        for ply in range(0, 400, 7):
            for in_check in (False, True):
                soft, hard, depth_cap = chess_agent.compute_time_budget(remaining, ply, in_check)
                usable = max(0.0, remaining - chess_agent.SAFETY_RESERVE_S
                             - chess_agent.PER_MOVE_OVERHEAD_S)
                assert soft <= hard <= usable * chess_agent.HARD_FRACTION + 1e-12


def test_panic_budget_is_tiny() -> None:
    soft, hard, depth_cap = chess_agent.compute_time_budget(0.5, 100, False)
    assert hard <= chess_agent.PANIC_LIMIT_S
    assert depth_cap == chess_agent.PANIC_DEPTH_CAP


@pytest.mark.parametrize("core", ["A", "B"])
def test_mate_in_two_is_found(core: str) -> None:
    board = chess.Board(MATE_IN_TWO_FEN)
    start = time.perf_counter()
    if core == "B":
        engine = chess_agent.MailboxEngine(mailbox_from_fen(MATE_IN_TWO_FEN))
        move, score, depth = engine.think(2.0, 2.0, 6, start)
        uci = chess_agent.move_to_uci(move)
    else:
        engine = chess_agent.PythonChessEngine()
        keys = [chess_agent.position_key(board)]
        move, score, depth = engine.think(board, keys, 2.0, 2.0, 6, start)
        uci = move.uci()
    assert score == chess_agent.MATE_VALUE - 3
    board.push_uci(uci)
    for reply in list(board.legal_moves):
        board.push(reply)
        mates = False
        for finisher in list(board.legal_moves):
            board.push(finisher)
            mates = mates or board.is_checkmate()
            board.pop()
        board.pop()
        assert mates


@pytest.mark.parametrize("core", ["A", "B"])
def test_wins_hanging_queen(core: str) -> None:
    board = chess.Board(HANGING_QUEEN_FEN)
    start = time.perf_counter()
    if core == "B":
        engine = chess_agent.MailboxEngine(mailbox_from_fen(HANGING_QUEEN_FEN))
        move, score, depth = engine.think(0.5, 0.5, 4, start)
        uci = chess_agent.move_to_uci(move)
    else:
        engine = chess_agent.PythonChessEngine()
        keys = [chess_agent.position_key(board)]
        move, score, depth = engine.think(board, keys, 0.5, 0.5, 4, start)
        uci = move.uci()
    assert uci == "c3d5"


def test_act_returns_legal_move_when_search_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_search(observation: dict, start_time: float, remaining: float) -> str:
        """Stand in for the search and always fail."""
        raise RuntimeError("forced failure")

    monkeypatch.setattr(chess_agent.STATE, "search_move", broken_search)
    board = chess.Board(CHECK_POSITIONS[2])
    observation = make_observation(board)
    observation["ply"] = 40
    result = chess_agent.act(observation)
    assert result["move"] in observation["legal_moves"]


def test_act_survives_garbage_observation() -> None:
    result = chess_agent.act({"legal_moves": ["e2e4", "d2d4"], "fen": "not a fen"})
    assert result["move"] in ("e2e4", "d2d4")


def test_act_plays_only_move_instantly() -> None:
    board = chess.Board("7k/8/8/8/8/8/6q1/7K w - - 0 1")
    observation = make_observation(board)
    assert observation["legal_moves"] == ["h1g2"]
    assert chess_agent.act(observation) == {"move": "h1g2"}


def test_act_uses_book_at_start() -> None:
    observation = make_observation(chess.Board())
    assert chess_agent.act(observation)["move"] in ("e2e4", "d2d4")


def test_act_low_clock_is_fast_and_legal() -> None:
    board = chess.Board(CHECK_POSITIONS[2])
    observation = make_observation(board, time_remaining=0.2)
    start = time.perf_counter()
    result = chess_agent.act(observation)
    assert time.perf_counter() - start < 0.05
    assert result["move"] in observation["legal_moves"]


def test_act_plays_a_full_game_against_itself() -> None:
    board = chess.Board()
    clocks = {chess.WHITE: 5.0, chess.BLACK: 5.0}
    while not board.is_game_over(claim_draw=True) and board.ply() < 120:
        observation = make_observation(board, clocks[board.turn])
        start = time.perf_counter()
        move = chess_agent.act(observation)["move"]
        clocks[board.turn] -= time.perf_counter() - start
        assert move in observation["legal_moves"]
        assert clocks[board.turn] > 0.3
        board.push_uci(move)


def test_sync_resets_for_a_new_game() -> None:
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        board.push_uci(uci)
    chess_agent.STATE.sync(make_observation(board))
    assert len(chess_agent.STATE.known_history) == 3
    other = chess.Board()
    other.push_uci("d2d4")
    chess_agent.STATE.sync(make_observation(other))
    assert chess_agent.STATE.known_history == ["d2d4"]
    assert chess_agent.STATE.mailbox.matches(chess_agent.STATE.board)


def test_sync_falls_back_to_fen_without_history() -> None:
    board = chess.Board(CHECK_POSITIONS[3])
    observation = make_observation(board)
    observation["history"] = ["e2e4"]
    chess_agent.STATE.sync(observation)
    assert chess_agent.fen_fields_match(chess_agent.STATE.board, observation["fen"])
    assert chess_agent.STATE.core_b_enabled
    assert chess_agent.STATE.mailbox.matches(chess_agent.STATE.board)


def test_core_b_is_disabled_on_root_mismatch() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    observation = make_observation(board)
    chess_agent.STATE.sync(observation)
    assert not chess_agent.STATE.core_b_root_matches(observation["legal_moves"] + ["a1a1"])
    assert not chess_agent.STATE.core_b_enabled
    observation["ply"] = 30
    assert chess_agent.act(observation)["move"] in observation["legal_moves"]

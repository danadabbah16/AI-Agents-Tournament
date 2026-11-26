"""Chess agent: PeSTO evaluation searched by alpha-beta over python-chess.

Started from: student_templates/chess_agent.py
Rules: games/chess/README.md
Core A searches python-chess boards directly. The Core B mailbox board below is
built and tested here; act() does not use it yet.
"""
import math
import random
import time

import chess

DEBUG_PRINTS = False

SAFETY_RESERVE_S = 0.5
PER_MOVE_OVERHEAD_S = 0.005
PANIC_THRESHOLD_S = 0.6
INSTANT_THRESHOLD_S = 0.25
PANIC_LIMIT_S = 0.02
PANIC_FRACTION = 0.05
PANIC_DEPTH_CAP = 2
EXPECTED_GAME_MOVES = 45
MIN_MOVES_TO_GO = 18
MAX_PLIES = 400
HARD_FRACTION = 0.2
HARD_MULTIPLIER = 3.0
CHECK_TIME_FACTOR = 1.3
OPENING_TIME_FACTOR = 1.2
TIME_CHECK_MASK = 63
ITERATION_START_FRACTION = 0.5
SCORE_DROP_MARGIN = 40
DEFAULT_CLOCK_S = 5.0

MATE_VALUE = 30000
MATE_BOUND = 29000
INFINITY_SCORE = 32000
MAX_PLY = 64
MAX_SEARCH_DEPTH = 32
DELTA_MARGIN = 200
BIG_DELTA = 975
DELTA_ENDGAME_PHASE = 4
HISTORY_MAX = 400000
TT_MAX_ENTRIES = 200000
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
TEMPO_BONUS = 10
TERMINAL_MATE_SCORE = 100000

TT_FLAG_SHIFT = 18
TT_DEPTH_SHIFT = 20
TT_AGE_SHIFT = 27
TT_SCORE_SHIFT = 35
TT_DEPTH_MASK = 127
TT_AGE_MASK = 255
TT_SCORE_OFFSET = 65536

TT_MOVE_SCORE = 4000000
QUEEN_PROMOTION_SCORE = 3000000
GOOD_CAPTURE_SCORE = 2000000
FIRST_KILLER_SCORE = 1900000
SECOND_KILLER_SCORE = 1800000
LOSING_CAPTURE_SCORE = 100000

BOOK_MAX_PLY = 24
ZOBRIST_SEED = 20261126

BISHOP_PAIR_MG = 30
BISHOP_PAIR_EG = 50
FULL_PHASE = 24
CONTEMPT_BASE = 10
CONTEMPT_PHASE_BONUS = 20
LOSING_THRESHOLD = -150
LOST_THRESHOLD = -400
LOST_CONTEMPT = -30

MOP_UP_CENTER_WEIGHT = 47
MOP_UP_KING_WEIGHT = 16
MOP_UP_CORNER_WEIGHT = 35
MOP_UP_MAX_DISTANCE = 14

EMPTY = 0
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6
WHITE_PAWN = 1
WHITE_KNIGHT = 2
WHITE_BISHOP = 3
WHITE_ROOK = 4
WHITE_QUEEN = 5
WHITE_KING = 6
BLACK_PAWN = 9
BLACK_KNIGHT = 10
BLACK_BISHOP = 11
BLACK_ROOK = 12
BLACK_QUEEN = 13
BLACK_KING = 14
OFF_BOARD = 32
BLACK_OFFSET = 8
PIECE_CODE_COUNT = 16

WHITE_SIDE = 0
BLACK_SIDE = 1

FLAG_QUIET = 0
FLAG_DOUBLE_PUSH = 1
FLAG_KING_CASTLE = 2
FLAG_QUEEN_CASTLE = 3
FLAG_CAPTURE = 4
FLAG_EN_PASSANT = 5
FLAG_PROMOTION = 8
FLAG_CAPTURE_PROMOTION = 12
FLAG_QUEEN_PROMOTION = 11
FLAG_QUEEN_CAPTURE_PROMOTION = 15
MOVE_MASK = (1 << 18) - 1
TO_SHIFT = 7
FLAG_SHIFT = 14
SQUARE_MASK = 127
PROMOTION_LETTERS = "nbrq"

CASTLE_WHITE_KING = 1
CASTLE_WHITE_QUEEN = 2
CASTLE_BLACK_KING = 4
CASTLE_BLACK_QUEEN = 8

KNIGHT_OFFSETS = (-21, -19, -12, -8, 8, 12, 19, 21)
BISHOP_DIRECTIONS = (-11, -9, 9, 11)
ROOK_DIRECTIONS = (-10, -1, 1, 10)
KING_OFFSETS = (-11, -10, -9, -1, 1, 9, 10, 11)
SLIDER_DIRECTIONS = {
    BISHOP: BISHOP_DIRECTIONS,
    ROOK: ROOK_DIRECTIONS,
    QUEEN: KING_OFFSETS,
}

MG_VALUES = [0, 82, 337, 365, 477, 1025, 0]
EG_VALUES = [0, 94, 281, 297, 512, 936, 0]
PHASE_INCREMENTS = [0, 0, 1, 1, 2, 4, 0]
SEE_VALUES = [0, 100, 320, 330, 500, 900, 20000]

MG_PAWN_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    98, 134, 61, 95, 68, 126, 34, -11,
    -6, 7, 26, 31, 65, 56, 25, -20,
    -14, 13, 6, 21, 23, 12, 17, -23,
    -27, -2, -5, 12, 17, 6, 10, -25,
    -26, -4, -4, -10, 3, 3, 33, -12,
    -35, -1, -20, -23, -15, 24, 38, -22,
    0, 0, 0, 0, 0, 0, 0, 0,
]
EG_PAWN_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    178, 173, 158, 134, 147, 132, 165, 187,
    94, 100, 85, 67, 56, 53, 82, 84,
    32, 24, 13, 5, -2, 4, 17, 17,
    13, 9, -3, -7, -7, -8, 3, -1,
    4, 7, -6, 1, 0, -5, -1, -8,
    13, 8, 8, 10, 13, 0, 2, -7,
    0, 0, 0, 0, 0, 0, 0, 0,
]
MG_KNIGHT_TABLE = [
    -167, -89, -34, -49, 61, -97, -15, -107,
    -73, -41, 72, 36, 23, 62, 7, -17,
    -47, 60, 37, 65, 84, 129, 73, 44,
    -9, 17, 19, 53, 37, 69, 18, 22,
    -13, 4, 16, 13, 28, 19, 21, -8,
    -23, -9, 12, 10, 19, 17, 25, -16,
    -29, -53, -12, -3, -1, 18, -14, -19,
    -105, -21, -58, -33, -17, -28, -19, -23,
]
EG_KNIGHT_TABLE = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25, -8, -25, -2, -9, -25, -24, -52,
    -24, -20, 10, 9, -1, -9, -19, -41,
    -17, 3, 22, 22, 22, 11, 8, -18,
    -18, -6, 16, 25, 16, 17, 4, -18,
    -23, -3, -1, 15, 10, -3, -20, -22,
    -42, -20, -10, -5, -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
]
MG_BISHOP_TABLE = [
    -29, 4, -82, -37, -25, -42, 7, -8,
    -26, 16, -18, -13, 30, 59, 18, -47,
    -16, 37, 43, 40, 35, 50, 37, -2,
    -4, 5, 19, 50, 37, 37, 7, -2,
    -6, 13, 13, 26, 34, 12, 10, 4,
    0, 15, 15, 15, 14, 27, 18, 10,
    4, 15, 16, 0, 7, 21, 33, 1,
    -33, -3, -14, -21, -13, -12, -39, -21,
]
EG_BISHOP_TABLE = [
    -14, -21, -11, -8, -7, -9, -17, -24,
    -8, -4, 7, -12, -3, -13, -4, -14,
    2, -8, 0, -1, -2, 6, 0, 4,
    -3, 9, 12, 9, 14, 10, 3, 2,
    -6, 3, 13, 19, 7, 10, -3, -9,
    -12, -3, 8, 10, 13, 3, -7, -15,
    -14, -18, -7, -1, 4, -9, -15, -27,
    -23, -9, -23, -5, -9, -16, -5, -17,
]
MG_ROOK_TABLE = [
    32, 42, 32, 51, 63, 9, 31, 43,
    27, 32, 58, 62, 80, 67, 26, 44,
    -5, 19, 26, 36, 17, 45, 61, 16,
    -24, -11, 7, 26, 24, 35, -8, -20,
    -36, -26, -12, -1, 9, -7, 6, -23,
    -45, -25, -16, -17, 3, 0, -5, -33,
    -44, -16, -20, -9, -1, 11, -6, -71,
    -19, -13, 1, 17, 16, 7, -37, -26,
]
EG_ROOK_TABLE = [
    13, 10, 18, 15, 12, 12, 8, 5,
    11, 13, 13, 11, -3, 3, 8, 3,
    7, 7, 7, 5, 4, -3, -5, -3,
    4, 3, 13, 1, 2, 1, -1, 2,
    3, 5, 8, 4, -5, -6, -8, -11,
    -4, 0, -5, -1, -7, -12, -8, -16,
    -6, -6, 0, 2, -9, -9, -11, -3,
    -9, 2, 3, -1, -5, -13, 4, -20,
]
MG_QUEEN_TABLE = [
    -28, 0, 29, 12, 59, 44, 43, 45,
    -24, -39, -5, 1, -16, 57, 28, 54,
    -13, -17, 7, 8, 29, 56, 47, 57,
    -27, -27, -16, -16, -1, 17, -2, 1,
    -9, -26, -9, -10, -2, -4, 3, -3,
    -14, 2, -11, -2, -5, 2, 14, 5,
    -35, -8, 11, 2, 8, 15, -3, 1,
    -1, -18, -9, 10, -15, -25, -31, -50,
]
EG_QUEEN_TABLE = [
    -9, 22, 22, 27, 27, 19, 10, 20,
    -17, 20, 32, 41, 58, 25, 30, 0,
    -20, 6, 9, 49, 47, 35, 19, 9,
    3, 22, 24, 45, 57, 40, 57, 36,
    -18, 28, 19, 47, 31, 34, 39, 23,
    -16, -27, 15, 6, 9, 17, 10, 5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43, -5, -32, -20, -41,
]
MG_KING_TABLE = [
    -65, 23, 16, -15, -56, -34, 2, 13,
    29, -1, -20, -7, -8, -4, -38, -29,
    -9, 24, 2, -16, -20, 6, 22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49, -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
    1, 7, -8, -64, -43, -16, 9, 8,
    -15, 36, 12, -54, 8, -28, 24, 14,
]
EG_KING_TABLE = [
    -74, -35, -18, -18, -11, 15, 4, -17,
    -12, 17, 14, 17, 17, 38, 23, 11,
    10, 17, 23, 15, 20, 45, 44, 13,
    -8, 22, 24, 27, 26, 33, 26, 3,
    -18, -4, 21, 24, 27, 23, 9, -11,
    -19, -3, 11, 21, 23, 16, 7, -9,
    -27, -11, 4, 13, 14, 4, -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
]
CENTER_MANHATTAN_DISTANCE = [
    6, 5, 4, 3, 3, 4, 5, 6,
    5, 4, 3, 2, 2, 3, 4, 5,
    4, 3, 2, 1, 1, 2, 3, 4,
    3, 2, 1, 0, 0, 1, 2, 3,
    3, 2, 1, 0, 0, 1, 2, 3,
    4, 3, 2, 1, 1, 2, 3, 4,
    5, 4, 3, 2, 2, 3, 4, 5,
    6, 5, 4, 3, 3, 4, 5, 6,
]
MG_TABLES = [None, MG_PAWN_TABLE, MG_KNIGHT_TABLE, MG_BISHOP_TABLE,
             MG_ROOK_TABLE, MG_QUEEN_TABLE, MG_KING_TABLE]
EG_TABLES = [None, EG_PAWN_TABLE, EG_KNIGHT_TABLE, EG_BISHOP_TABLE,
             EG_ROOK_TABLE, EG_QUEEN_TABLE, EG_KING_TABLE]


def build_square_maps() -> tuple:
    """Build the 64-to-120 and 120-to-64 square conversion tables."""
    square_120_from_64 = []
    square_64_from_120 = [-1] * 120
    for square in range(64):
        square_120 = 21 + (square % 8) + 10 * (square // 8)
        square_120_from_64.append(square_120)
        square_64_from_120[square_120] = square
    return tuple(square_120_from_64), tuple(square_64_from_120)


SQUARE_120_FROM_64, SQUARE_64_FROM_120 = build_square_maps()
BOARD_SQUARES_120 = SQUARE_120_FROM_64


def build_merged_tables() -> tuple:
    """Merge material and PST into tables by piece code and square, Black negated."""
    mg_64 = [0] * (PIECE_CODE_COUNT * 64)
    eg_64 = [0] * (PIECE_CODE_COUNT * 64)
    mg_120 = [0] * (PIECE_CODE_COUNT * 120)
    eg_120 = [0] * (PIECE_CODE_COUNT * 120)
    for piece_type in range(PAWN, KING + 1):
        mg_table = MG_TABLES[piece_type]
        eg_table = EG_TABLES[piece_type]
        black_code = piece_type + BLACK_OFFSET
        for square in range(64):
            # The tables are drawn from White's side with a8 first, so White flips ranks.
            white_mg = MG_VALUES[piece_type] + mg_table[square ^ 56]
            white_eg = EG_VALUES[piece_type] + eg_table[square ^ 56]
            black_mg = -(MG_VALUES[piece_type] + mg_table[square])
            black_eg = -(EG_VALUES[piece_type] + eg_table[square])
            square_120 = SQUARE_120_FROM_64[square]
            mg_64[piece_type * 64 + square] = white_mg
            eg_64[piece_type * 64 + square] = white_eg
            mg_64[black_code * 64 + square] = black_mg
            eg_64[black_code * 64 + square] = black_eg
            mg_120[piece_type * 120 + square_120] = white_mg
            eg_120[piece_type * 120 + square_120] = white_eg
            mg_120[black_code * 120 + square_120] = black_mg
            eg_120[black_code * 120 + square_120] = black_eg
    return mg_64, eg_64, mg_120, eg_120


MG_64, EG_64, MG_120, EG_120 = build_merged_tables()


def build_phase_table() -> list:
    """Return the game-phase weight of every piece code."""
    phase_of_code = [0] * PIECE_CODE_COUNT
    for piece_type in range(PAWN, KING + 1):
        phase_of_code[piece_type] = PHASE_INCREMENTS[piece_type]
        phase_of_code[piece_type + BLACK_OFFSET] = PHASE_INCREMENTS[piece_type]
    return phase_of_code


PHASE_OF_CODE = build_phase_table()


def build_castling_mask() -> list:
    """Return the per-square mask that clears castling rights when touched."""
    mask = [15] * 120
    mask[21] = 13
    mask[25] = 12
    mask[28] = 14
    mask[91] = 7
    mask[95] = 3
    mask[98] = 11
    return mask


CASTLING_MASK = build_castling_mask()


def build_zobrist_keys() -> tuple:
    """Create reproducible 64-bit Zobrist keys from a private generator."""
    # A private generator keeps the keys identical whatever seed the judge sets.
    generator = random.Random(ZOBRIST_SEED)
    piece_keys = [0] * (PIECE_CODE_COUNT * 120)
    for piece_code in range(PIECE_CODE_COUNT):
        for square_120 in range(120):
            piece_keys[piece_code * 120 + square_120] = generator.getrandbits(64)
    castling_keys = []
    for castling in range(16):
        castling_keys.append(generator.getrandbits(64))
    ep_keys = []
    for file_index in range(8):
        ep_keys.append(generator.getrandbits(64))
    side_key = generator.getrandbits(64)
    return piece_keys, castling_keys, ep_keys, side_key


ZOBRIST_PIECES, ZOBRIST_CASTLING, ZOBRIST_EP_FILES, ZOBRIST_SIDE = build_zobrist_keys()


BOOK_LINES = [
    ("e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 "
     "c2c3 e8g8 h2h3", 30),
    ("e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d3 d7d6 e1g1 e8g8 f1e1", 25),
    ("e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5 d4b3 c8e6 "
     "f2f3", 25),
    ("e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 f1e2 e7e5 d4b3 f8e7", 15),
    ("e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 d7d6 c1g5 e7e6 d1d2", 20),
    ("e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 g8f6 b1c3 d7d6", 10),
    ("e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5 e4g3 f5g6 h2h4 h7h6 g1f3 b8d7 "
     "h4h5 g6h7 f1d3 h7d3 d1d3", 25),
    ("e2e4 c7c6 d2d4 d7d5 e4e5 c8f5 g1f3 e7e6 f1e2", 10),
    ("e2e4 c7c6 d2d4 d7d5 e4d5 c6d5 f1d3 b8c6 c2c3 g8f6", 8),
    ("e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 c1g5 f8e7 e4e5 f6d7 g5e7 d8e7", 15),
    ("d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 g1f3 b8d7 a1c1", 25),
    ("d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4 a2a4 c8f5 e2e3 e7e6 f1c4", 20),
    ("d2d4 d7d5 c2c4 d5c4 g1f3 g8f6 e2e3 e7e6 f1c4 c7c5 e1g1 a7a6", 10),
    ("d2d4 d7d5 g1f3 g8f6 c2c4 e7e6 b1c3 f8e7", 10),
    ("d2d4 d7d5 c1f4 g8f6 e2e3 c7c5 g1f3 b8c6", 8),
    ("d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2 e7e5 e1g1", 15),
    ("d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 d1c2 e8g8 a2a3 b4c3 c2c3", 15),
    ("c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 d7d5 c4d5 f6d5 f1g2 d5b6", 15),
    ("g1f3 d7d5 g2g3 g8f6 f1g2 e7e6 e1g1 f8e7 d2d3 e8g8", 15),
]
WHITE_FIRST_MOVES = [["e2e4", 55], ["d2d4", 45]]


class SearchTimeout(Exception):
    """Raised inside a search when the hard deadline has passed."""


def debug_print(message: str) -> None:
    """Print a message only while DEBUG_PRINTS is on."""
    if DEBUG_PRINTS:
        print(message)


def book_key(board: chess.Board) -> str:
    """Return the first four FEN fields, the key used by the opening book."""
    fields = board.fen().split(" ")
    return " ".join(fields[:4])


def add_book_entry(book: dict, key: str, uci: str, weight: int) -> None:
    """Add a book move, keeping the larger weight when it is already listed."""
    entries = book.setdefault(key, [])
    for entry in entries:
        if entry[0] == uci:
            if weight > entry[1]:
                entry[1] = weight
            return
    entries.append([uci, weight])


def build_opening_book() -> dict:
    """Replay every book line and index its moves by the position before them."""
    book = {}
    for line, weight in BOOK_LINES:
        board = chess.Board()
        for uci in line.split():
            try:
                move = chess.Move.from_uci(uci)
            except ValueError as error:
                debug_print(f"Book line stopped at {uci}: {error}")
                break
            if not board.is_legal(move):
                debug_print(f"Book line stopped at illegal move {uci}")
                break
            add_book_entry(book, book_key(board), uci, weight)
            board.push(move)
    start_entries = []
    for uci, weight in WHITE_FIRST_MOVES:
        start_entries.append([uci, weight])
    book[book_key(chess.Board())] = start_entries
    return book


OPENING_BOOK = build_opening_book()


def pick_weighted(entries: list) -> str:
    """Pick a book move with probability proportional to its weight."""
    total = 0
    for entry in entries:
        total += entry[1]
    # The judge seeds the global generator, so choices differ between games.
    roll = random.random() * total
    running = 0
    for entry in entries:
        running += entry[1]
        if roll < running:
            return entry[0]
    return entries[-1][0]


def manhattan_distance(first_square: int, second_square: int) -> int:
    """Return the Manhattan distance between two 0-63 squares."""
    file_gap = abs((first_square & 7) - (second_square & 7))
    rank_gap = abs((first_square >> 3) - (second_square >> 3))
    return file_gap + rank_gap


def mop_up_bonus(loser_king: int, winner_king: int, knights: int, bishops: int,
                 rooks: int, queens: int, bishop_square: int) -> int:
    """Return the bonus for driving a bare king to the edge, or 0 if mate is impossible."""
    can_mate = rooks > 0 or queens > 0 or bishops >= 2 or (bishops >= 1 and knights >= 1)
    if not can_mate:
        return 0
    king_distance = manhattan_distance(loser_king, winner_king)
    bonus = MOP_UP_KING_WEIGHT * (MOP_UP_MAX_DISTANCE - king_distance)
    is_bishop_knight = rooks == 0 and queens == 0 and bishops == 1 and knights == 1
    if is_bishop_knight and bishop_square >= 0:
        # KBNK can only mate in a corner of the bishop's colour.
        if ((bishop_square & 7) + (bishop_square >> 3)) % 2 == 0:
            corners = (0, 63)
        else:
            corners = (7, 56)
        corner_distance = min(manhattan_distance(loser_king, corners[0]),
                              manhattan_distance(loser_king, corners[1]))
        return bonus + MOP_UP_CORNER_WEIGHT * (MOP_UP_MAX_DISTANCE - corner_distance)
    return bonus + MOP_UP_CENTER_WEIGHT * CENTER_MANHATTAN_DISTANCE[loser_king]


def compute_contempt(phase: int, our_static: int) -> int:
    """Return the draw penalty for our side, smaller in endgames and when losing."""
    if our_static < LOST_THRESHOLD:
        return LOST_CONTEMPT
    if our_static < LOSING_THRESHOLD:
        return 0
    capped_phase = min(phase, FULL_PHASE)
    return CONTEMPT_BASE + (CONTEMPT_PHASE_BONUS * capped_phase) // FULL_PHASE


def new_killer_table() -> list:
    """Return two empty killer slots for every ply."""
    killers = []
    for ply in range(MAX_PLY + 2):
        killers.append([0, 0])
    return killers


def pack_tt_entry(move: int, flag: int, depth: int, age: int, score: int) -> int:
    """Pack a transposition entry into one integer."""
    # Integers are not tracked by the garbage collector, unlike tuples.
    return (move | (flag << TT_FLAG_SHIFT) | (min(depth, TT_DEPTH_MASK) << TT_DEPTH_SHIFT)
            | ((age & TT_AGE_MASK) << TT_AGE_SHIFT)
            | ((score + TT_SCORE_OFFSET) << TT_SCORE_SHIFT))


def store_tt_entry(table: dict, key, move: int, flag: int, depth: int, age: int,
                   score: int, ply: int) -> None:
    """Store a search result, replacing only shallower, older or non-exact entries."""
    if score > MATE_BOUND:
        score += ply
    elif score < -MATE_BOUND:
        score -= ply
    old_entry = table.get(key)
    if old_entry is not None and flag != TT_EXACT:
        old_depth = (old_entry >> TT_DEPTH_SHIFT) & TT_DEPTH_MASK
        old_age = (old_entry >> TT_AGE_SHIFT) & TT_AGE_MASK
        if old_depth > depth and old_age == (age & TT_AGE_MASK):
            return
    table[key] = pack_tt_entry(move, flag, depth, age, score)


def probe_tt_score(entry: int, depth: int, alpha: int, beta: int, ply: int):
    """Return the stored score if it settles this node, otherwise None."""
    if ((entry >> TT_DEPTH_SHIFT) & TT_DEPTH_MASK) < depth:
        return None
    score = (entry >> TT_SCORE_SHIFT) - TT_SCORE_OFFSET
    if score > MATE_BOUND:
        score -= ply
    elif score < -MATE_BOUND:
        score += ply
    flag = (entry >> TT_FLAG_SHIFT) & 3
    if flag == TT_EXACT:
        return score
    if flag == TT_LOWER and score >= beta:
        return score
    if flag == TT_UPPER and score <= alpha:
        return score
    return None


def trim_tt(table: dict) -> dict:
    """Return the table with its oldest half dropped once it grows too large."""
    if len(table) <= TT_MAX_ENTRIES:
        return table
    # Dicts keep insertion order, so the first keys are the oldest ones.
    skip = len(table) - TT_MAX_ENTRIES // 2
    newest = {}
    index = 0
    for key, entry in table.items():
        if index >= skip:
            newest[key] = entry
        index += 1
    return newest


def halve_history(history: list) -> None:
    """Halve every history score so old cutoffs fade."""
    for i in range(len(history)):
        history[i] >>= 1


def square_name_120(square_120: int) -> str:
    """Return the algebraic name of a 120-board square."""
    file_index = (square_120 % 10) - 1
    rank_index = (square_120 // 10) - 2
    return "abcdefgh"[file_index] + str(rank_index + 1)


def move_to_uci(move: int) -> str:
    """Convert a packed Core B move to UCI, including the promotion letter."""
    from_square = move & SQUARE_MASK
    to_square = (move >> TO_SHIFT) & SQUARE_MASK
    flag = move >> FLAG_SHIFT
    uci = square_name_120(from_square) + square_name_120(to_square)
    if flag & FLAG_PROMOTION:
        uci += PROMOTION_LETTERS[flag & 3]
    return uci


def piece_code_of(piece: chess.Piece) -> int:
    """Return the Core B piece code of a python-chess piece."""
    if piece.color == chess.WHITE:
        return piece.piece_type
    return piece.piece_type + BLACK_OFFSET


def castling_bits_of(board: chess.Board) -> int:
    """Return python-chess castling rights as the Core B bitmask."""
    rights = board.clean_castling_rights()
    bits = 0
    if rights & chess.BB_H1:
        bits |= CASTLE_WHITE_KING
    if rights & chess.BB_A1:
        bits |= CASTLE_WHITE_QUEEN
    if rights & chess.BB_H8:
        bits |= CASTLE_BLACK_KING
    if rights & chess.BB_A8:
        bits |= CASTLE_BLACK_QUEEN
    return bits


class MailboxBoard:
    """Core B position: a 10x12 mailbox with in-place make and unmake."""

    def __init__(self) -> None:
        """Start from the standard initial position."""
        self.squares = [OFF_BOARD] * 120
        self.side = WHITE_SIDE
        self.castling = 0
        self.ep_square = 0
        self.ep_hash_key = 0
        self.halfmove = 0
        self.king_squares = [0, 0]
        self.piece_counts = [0] * PIECE_CODE_COUNT
        self.mg_score = 0
        self.eg_score = 0
        self.phase = 0
        self.hash_value = 0
        self.hash_history = []
        self.undo_stack = []
        self.load_board(chess.Board())

    def load_board(self, board: chess.Board) -> None:
        """Copy a python-chess position, starting a fresh hash history."""
        squares = [OFF_BOARD] * 120
        for square_120 in BOARD_SQUARES_120:
            squares[square_120] = EMPTY
        for square, piece in board.piece_map().items():
            squares[SQUARE_120_FROM_64[square]] = piece_code_of(piece)
        self.squares = squares
        self.side = WHITE_SIDE if board.turn == chess.WHITE else BLACK_SIDE
        self.castling = castling_bits_of(board)
        if board.ep_square is None:
            self.ep_square = 0
        else:
            self.ep_square = SQUARE_120_FROM_64[board.ep_square]
        self.halfmove = board.halfmove_clock
        self.undo_stack = []
        self.refresh_derived_state()
        self.hash_history = [self.hash_value]

    def refresh_derived_state(self) -> None:
        """Recompute counts, kings, scores, phase and hash from the squares."""
        counts, kings, mg_score, eg_score, phase = self.compute_material()
        self.piece_counts = counts
        self.king_squares = kings
        self.mg_score = mg_score
        self.eg_score = eg_score
        self.phase = phase
        self.ep_hash_key = self.compute_ep_hash_key()
        self.hash_value = self.compute_hash()

    def compute_material(self) -> tuple:
        """Return piece counts, king squares and scores computed from scratch."""
        counts = [0] * PIECE_CODE_COUNT
        kings = [0, 0]
        mg_score = 0
        eg_score = 0
        phase = 0
        for square_120 in BOARD_SQUARES_120:
            piece = self.squares[square_120]
            if piece == EMPTY:
                continue
            counts[piece] += 1
            mg_score += MG_120[piece * 120 + square_120]
            eg_score += EG_120[piece * 120 + square_120]
            phase += PHASE_OF_CODE[piece]
            if piece == WHITE_KING:
                kings[WHITE_SIDE] = square_120
            elif piece == BLACK_KING:
                kings[BLACK_SIDE] = square_120
        return counts, kings, mg_score, eg_score, phase

    def compute_ep_hash_key(self) -> int:
        """Return the en-passant hash key, used only if a capture is possible."""
        if self.ep_square == 0:
            return 0
        if self.side == WHITE_SIDE:
            pushed_pawn = self.ep_square - 10
            capturer = WHITE_PAWN
        else:
            pushed_pawn = self.ep_square + 10
            capturer = BLACK_PAWN
        if self.squares[pushed_pawn - 1] == capturer or self.squares[pushed_pawn + 1] == capturer:
            return ZOBRIST_EP_FILES[(self.ep_square % 10) - 1]
        return 0

    def compute_hash(self) -> int:
        """Return the Zobrist hash computed from scratch."""
        hash_value = 0
        for square_120 in BOARD_SQUARES_120:
            piece = self.squares[square_120]
            if piece != EMPTY:
                hash_value ^= ZOBRIST_PIECES[piece * 120 + square_120]
        hash_value ^= ZOBRIST_CASTLING[self.castling]
        hash_value ^= self.compute_ep_hash_key()
        if self.side == BLACK_SIDE:
            hash_value ^= ZOBRIST_SIDE
        return hash_value

    def matches(self, board: chess.Board) -> bool:
        """Return True when this position and its incremental state equal the board."""
        for square in range(64):
            piece = board.piece_at(square)
            expected = EMPTY if piece is None else piece_code_of(piece)
            if self.squares[SQUARE_120_FROM_64[square]] != expected:
                return False
        side = WHITE_SIDE if board.turn == chess.WHITE else BLACK_SIDE
        if self.side != side or self.castling != castling_bits_of(board):
            return False
        expected_ep = 0 if board.ep_square is None else SQUARE_120_FROM_64[board.ep_square]
        if self.ep_square != expected_ep or self.halfmove != board.halfmove_clock:
            return False
        counts, kings, mg_score, eg_score, phase = self.compute_material()
        if counts != self.piece_counts or kings != self.king_squares:
            return False
        if mg_score != self.mg_score or eg_score != self.eg_score or phase != self.phase:
            return False
        if self.ep_hash_key != self.compute_ep_hash_key():
            return False
        return self.hash_value == self.compute_hash() and self.hash_history[-1] == self.hash_value

    def make_move(self, move: int) -> None:
        """Play a packed pseudo-legal move in place and push its undo record."""
        squares = self.squares
        side = self.side
        counts = self.piece_counts
        from_square = move & SQUARE_MASK
        to_square = (move >> TO_SHIFT) & SQUARE_MASK
        flag = move >> FLAG_SHIFT
        piece = squares[from_square]
        captured = squares[to_square]
        self.undo_stack.append((move, captured, self.castling, self.ep_square, self.ep_hash_key,
                                self.halfmove, self.hash_value, self.mg_score, self.eg_score,
                                self.phase))
        hash_value = self.hash_value ^ self.ep_hash_key ^ ZOBRIST_SIDE
        mg_score = self.mg_score
        eg_score = self.eg_score
        phase = self.phase
        index = piece * 120 + from_square
        mg_score -= MG_120[index]
        eg_score -= EG_120[index]
        hash_value ^= ZOBRIST_PIECES[index]
        halfmove = self.halfmove + 1
        if captured != EMPTY:
            index = captured * 120 + to_square
            mg_score -= MG_120[index]
            eg_score -= EG_120[index]
            hash_value ^= ZOBRIST_PIECES[index]
            phase -= PHASE_OF_CODE[captured]
            counts[captured] -= 1
            halfmove = 0
        placed = piece
        if flag & FLAG_PROMOTION:
            placed = (flag & 3) + KNIGHT + (side << 3)
            counts[piece] -= 1
            counts[placed] += 1
            phase += PHASE_OF_CODE[placed]
        squares[from_square] = EMPTY
        squares[to_square] = placed
        index = placed * 120 + to_square
        mg_score += MG_120[index]
        eg_score += EG_120[index]
        hash_value ^= ZOBRIST_PIECES[index]
        if flag == FLAG_EN_PASSANT:
            if side == WHITE_SIDE:
                victim_square = to_square - 10
            else:
                victim_square = to_square + 10
            victim = squares[victim_square]
            squares[victim_square] = EMPTY
            index = victim * 120 + victim_square
            mg_score -= MG_120[index]
            eg_score -= EG_120[index]
            hash_value ^= ZOBRIST_PIECES[index]
            counts[victim] -= 1
        elif flag == FLAG_KING_CASTLE or flag == FLAG_QUEEN_CASTLE:
            if flag == FLAG_KING_CASTLE:
                rook_from = from_square + 3
                rook_to = from_square + 1
            else:
                rook_from = from_square - 4
                rook_to = from_square - 1
            rook = squares[rook_from]
            squares[rook_from] = EMPTY
            squares[rook_to] = rook
            index = rook * 120 + rook_from
            mg_score -= MG_120[index]
            eg_score -= EG_120[index]
            hash_value ^= ZOBRIST_PIECES[index]
            index = rook * 120 + rook_to
            mg_score += MG_120[index]
            eg_score += EG_120[index]
            hash_value ^= ZOBRIST_PIECES[index]
        piece_type = piece & 7
        if piece_type == KING:
            self.king_squares[side] = to_square
        elif piece_type == PAWN:
            halfmove = 0
        castling = self.castling
        new_castling = castling & CASTLING_MASK[from_square] & CASTLING_MASK[to_square]
        if new_castling != castling:
            hash_value ^= ZOBRIST_CASTLING[castling] ^ ZOBRIST_CASTLING[new_castling]
            self.castling = new_castling
        ep_hash_key = 0
        if flag == FLAG_DOUBLE_PUSH:
            self.ep_square = (from_square + to_square) >> 1
            enemy_pawn = BLACK_PAWN if side == WHITE_SIDE else WHITE_PAWN
            # Polyglot rule: the key counts only if an enemy pawn could capture.
            if squares[to_square - 1] == enemy_pawn or squares[to_square + 1] == enemy_pawn:
                ep_hash_key = ZOBRIST_EP_FILES[(to_square % 10) - 1]
                hash_value ^= ep_hash_key
        else:
            self.ep_square = 0
        self.ep_hash_key = ep_hash_key
        self.halfmove = halfmove
        self.side = side ^ 1
        self.mg_score = mg_score
        self.eg_score = eg_score
        self.phase = phase
        self.hash_value = hash_value
        self.hash_history.append(hash_value)

    def unmake_move(self) -> None:
        """Take back the last move from its undo record without recomputing."""
        record = self.undo_stack.pop()
        self.hash_history.pop()
        move = record[0]
        side = self.side ^ 1
        self.side = side
        self.castling = record[2]
        self.ep_square = record[3]
        self.ep_hash_key = record[4]
        self.halfmove = record[5]
        self.hash_value = record[6]
        self.mg_score = record[7]
        self.eg_score = record[8]
        self.phase = record[9]
        if move == 0:
            return
        squares = self.squares
        counts = self.piece_counts
        captured = record[1]
        from_square = move & SQUARE_MASK
        to_square = (move >> TO_SHIFT) & SQUARE_MASK
        flag = move >> FLAG_SHIFT
        piece = squares[to_square]
        if flag & FLAG_PROMOTION:
            counts[piece] -= 1
            piece = PAWN + (side << 3)
            counts[piece] += 1
        squares[from_square] = piece
        squares[to_square] = captured
        if captured != EMPTY:
            counts[captured] += 1
        if flag == FLAG_EN_PASSANT:
            if side == WHITE_SIDE:
                squares[to_square - 10] = BLACK_PAWN
                counts[BLACK_PAWN] += 1
            else:
                squares[to_square + 10] = WHITE_PAWN
                counts[WHITE_PAWN] += 1
        elif flag == FLAG_KING_CASTLE:
            squares[from_square + 3] = squares[from_square + 1]
            squares[from_square + 1] = EMPTY
        elif flag == FLAG_QUEEN_CASTLE:
            squares[from_square - 4] = squares[from_square - 1]
            squares[from_square - 1] = EMPTY
        if (piece & 7) == KING:
            self.king_squares[side] = from_square

    def make_null_move(self) -> None:
        """Pass the turn; the halfmove clock is zeroed so repetition scans stop here."""
        self.undo_stack.append((0, EMPTY, self.castling, self.ep_square, self.ep_hash_key,
                                self.halfmove, self.hash_value, self.mg_score, self.eg_score,
                                self.phase))
        hash_value = self.hash_value ^ self.ep_hash_key ^ ZOBRIST_SIDE
        self.ep_square = 0
        self.ep_hash_key = 0
        self.halfmove = 0
        self.side ^= 1
        self.hash_value = hash_value
        self.hash_history.append(hash_value)

    def undo_to(self, stack_size: int) -> None:
        """Unmake moves until the undo stack is back to the given size."""
        while len(self.undo_stack) > stack_size:
            self.unmake_move()

    def is_attacked(self, square: int, by_side: int) -> bool:
        """Return True if the given side attacks the square."""
        squares = self.squares
        if by_side == WHITE_SIDE:
            if squares[square - 9] == WHITE_PAWN or squares[square - 11] == WHITE_PAWN:
                return True
            knight = WHITE_KNIGHT
            bishop = WHITE_BISHOP
            rook = WHITE_ROOK
            queen = WHITE_QUEEN
            king = WHITE_KING
        else:
            if squares[square + 9] == BLACK_PAWN or squares[square + 11] == BLACK_PAWN:
                return True
            knight = BLACK_KNIGHT
            bishop = BLACK_BISHOP
            rook = BLACK_ROOK
            queen = BLACK_QUEEN
            king = BLACK_KING
        for offset in KNIGHT_OFFSETS:
            if squares[square + offset] == knight:
                return True
        for direction in BISHOP_DIRECTIONS:
            target_square = square + direction
            target = squares[target_square]
            if target == king:
                return True
            while target == EMPTY:
                target_square += direction
                target = squares[target_square]
            if target == bishop or target == queen:
                return True
        for direction in ROOK_DIRECTIONS:
            target_square = square + direction
            target = squares[target_square]
            if target == king:
                return True
            while target == EMPTY:
                target_square += direction
                target = squares[target_square]
            if target == rook or target == queen:
                return True
        return False

    def side_in_check(self) -> bool:
        """Return True if the side to move is in check."""
        return self.is_attacked(self.king_squares[self.side], self.side ^ 1)

    def generate_moves(self, captures_only: bool) -> list:
        """Return pseudo-legal packed moves; captures_only keeps captures and queen promotions."""
        squares = self.squares
        side = self.side
        ep_square = self.ep_square
        moves = []
        # Local alias avoids an attribute lookup per generated move.
        add_move = moves.append
        if side == WHITE_SIDE:
            enemy_low = BLACK_PAWN
            enemy_high = BLACK_KING
            forward = 10
            capture_directions = (9, 11)
            start_low = 31
            start_high = 38
            promotion_low = 91
            promotion_high = 98
        else:
            enemy_low = WHITE_PAWN
            enemy_high = WHITE_KING
            forward = -10
            capture_directions = (-9, -11)
            start_low = 81
            start_high = 88
            promotion_low = 21
            promotion_high = 28
        for from_square in BOARD_SQUARES_120:
            piece = squares[from_square]
            if piece == EMPTY or (piece >> 3) != side:
                continue
            piece_type = piece & 7
            if piece_type == PAWN:
                to_square = from_square + forward
                promotes = promotion_low <= to_square <= promotion_high
                base = from_square | (to_square << TO_SHIFT)
                if squares[to_square] == EMPTY:
                    if promotes:
                        add_move(base | (FLAG_QUEEN_PROMOTION << FLAG_SHIFT))
                        if not captures_only:
                            add_move(base | (FLAG_PROMOTION << FLAG_SHIFT))
                            add_move(base | ((FLAG_PROMOTION + 1) << FLAG_SHIFT))
                            add_move(base | ((FLAG_PROMOTION + 2) << FLAG_SHIFT))
                    elif not captures_only:
                        add_move(base)
                        double_square = to_square + forward
                        on_start_rank = start_low <= from_square <= start_high
                        if on_start_rank and squares[double_square] == EMPTY:
                            add_move(from_square | (double_square << TO_SHIFT)
                                     | (FLAG_DOUBLE_PUSH << FLAG_SHIFT))
                for direction in capture_directions:
                    to_square = from_square + direction
                    target = squares[to_square]
                    base = from_square | (to_square << TO_SHIFT)
                    if enemy_low <= target <= enemy_high:
                        if promotes:
                            add_move(base | (FLAG_QUEEN_CAPTURE_PROMOTION << FLAG_SHIFT))
                            if not captures_only:
                                add_move(base | (FLAG_CAPTURE_PROMOTION << FLAG_SHIFT))
                                add_move(base | ((FLAG_CAPTURE_PROMOTION + 1) << FLAG_SHIFT))
                                add_move(base | ((FLAG_CAPTURE_PROMOTION + 2) << FLAG_SHIFT))
                        else:
                            add_move(base | (FLAG_CAPTURE << FLAG_SHIFT))
                    elif to_square == ep_square:
                        add_move(base | (FLAG_EN_PASSANT << FLAG_SHIFT))
            elif piece_type == KNIGHT or piece_type == KING:
                offsets = KNIGHT_OFFSETS if piece_type == KNIGHT else KING_OFFSETS
                for offset in offsets:
                    to_square = from_square + offset
                    target = squares[to_square]
                    if target == EMPTY:
                        if not captures_only:
                            add_move(from_square | (to_square << TO_SHIFT))
                    elif enemy_low <= target <= enemy_high:
                        add_move(from_square | (to_square << TO_SHIFT)
                                 | (FLAG_CAPTURE << FLAG_SHIFT))
            else:
                for direction in SLIDER_DIRECTIONS[piece_type]:
                    to_square = from_square + direction
                    target = squares[to_square]
                    while target == EMPTY:
                        if not captures_only:
                            add_move(from_square | (to_square << TO_SHIFT))
                        to_square += direction
                        target = squares[to_square]
                    if enemy_low <= target <= enemy_high:
                        add_move(from_square | (to_square << TO_SHIFT)
                                 | (FLAG_CAPTURE << FLAG_SHIFT))
        if not captures_only and self.castling:
            self.add_castling_moves(moves)
        return moves

    def add_castling_moves(self, moves: list) -> None:
        """Append castling moves whose path is empty and not attacked."""
        squares = self.squares
        castling = self.castling
        if self.side == WHITE_SIDE:
            king_square = 25
            king_right = CASTLE_WHITE_KING
            queen_right = CASTLE_WHITE_QUEEN
            king = WHITE_KING
            rook = WHITE_ROOK
            enemy = BLACK_SIDE
        else:
            king_square = 95
            king_right = CASTLE_BLACK_KING
            queen_right = CASTLE_BLACK_QUEEN
            king = BLACK_KING
            rook = BLACK_ROOK
            enemy = WHITE_SIDE
        if squares[king_square] != king or self.is_attacked(king_square, enemy):
            return
        if (castling & king_right and squares[king_square + 1] == EMPTY
                and squares[king_square + 2] == EMPTY and squares[king_square + 3] == rook
                and not self.is_attacked(king_square + 1, enemy)
                and not self.is_attacked(king_square + 2, enemy)):
            moves.append(king_square | ((king_square + 2) << TO_SHIFT)
                         | (FLAG_KING_CASTLE << FLAG_SHIFT))
        if (castling & queen_right and squares[king_square - 1] == EMPTY
                and squares[king_square - 2] == EMPTY and squares[king_square - 3] == EMPTY
                and squares[king_square - 4] == rook
                and not self.is_attacked(king_square - 1, enemy)
                and not self.is_attacked(king_square - 2, enemy)):
            moves.append(king_square | ((king_square - 2) << TO_SHIFT)
                         | (FLAG_QUEEN_CASTLE << FLAG_SHIFT))

    def legal_moves(self) -> list:
        """Return the legal packed moves by filtering pseudo-legal ones."""
        side = self.side
        enemy = side ^ 1
        legal = []
        for move in self.generate_moves(False):
            self.make_move(move)
            if not self.is_attacked(self.king_squares[side], enemy):
                legal.append(move)
            self.unmake_move()
        return legal

    def parse_uci(self, uci: str) -> int:
        """Return the packed move for a UCI string, inferring flags from the board."""
        from_square = SQUARE_120_FROM_64[chess.parse_square(uci[0:2])]
        to_square = SQUARE_120_FROM_64[chess.parse_square(uci[2:4])]
        piece = self.squares[from_square]
        target = self.squares[to_square]
        piece_type = piece & 7
        flag = FLAG_QUIET
        if len(uci) > 4:
            flag = FLAG_PROMOTION + PROMOTION_LETTERS.index(uci[4].lower())
            if target != EMPTY:
                flag += FLAG_CAPTURE
        elif target != EMPTY:
            flag = FLAG_CAPTURE
        elif (piece_type == PAWN and to_square == self.ep_square
              and (to_square - from_square) % 10 != 0):
            flag = FLAG_EN_PASSANT
        elif piece_type == KING and to_square - from_square == 2:
            flag = FLAG_KING_CASTLE
        elif piece_type == KING and from_square - to_square == 2:
            flag = FLAG_QUEEN_CASTLE
        elif piece_type == PAWN and abs(to_square - from_square) == 20:
            flag = FLAG_DOUBLE_PUSH
        return from_square | (to_square << TO_SHIFT) | (flag << FLAG_SHIFT)

    def make_uci_move(self, uci: str) -> None:
        """Play a UCI move given by the judge."""
        self.make_move(self.parse_uci(uci))

    def is_repetition(self) -> bool:
        """Return True if this position already occurred since the last irreversible move."""
        history = self.hash_history
        current = history[-1]
        index = len(history) - 5
        stop = len(history) - 1 - self.halfmove
        if stop < 0:
            stop = 0
        while index >= stop:
            if history[index] == current:
                return True
            index -= 2
        return False

    def is_insufficient_material(self) -> bool:
        """Return True when neither side has mating material."""
        counts = self.piece_counts
        if counts[WHITE_PAWN] or counts[BLACK_PAWN] or counts[WHITE_ROOK] or counts[BLACK_ROOK]:
            return False
        if counts[WHITE_QUEEN] or counts[BLACK_QUEEN]:
            return False
        white_minors = counts[WHITE_KNIGHT] + counts[WHITE_BISHOP]
        black_minors = counts[BLACK_KNIGHT] + counts[BLACK_BISHOP]
        return white_minors <= 1 and black_minors <= 1

    def has_non_pawn_material(self) -> bool:
        """Return True if the side to move has a piece other than pawns and king."""
        counts = self.piece_counts
        offset = self.side << 3
        return (counts[KNIGHT + offset] + counts[BISHOP + offset] + counts[ROOK + offset]
                + counts[QUEEN + offset]) > 0

    def find_piece(self, piece_code: int) -> int:
        """Return the 0-63 square of the first piece with this code, or -1."""
        for square_120 in BOARD_SQUARES_120:
            if self.squares[square_120] == piece_code:
                return SQUARE_64_FROM_120[square_120]
        return -1

    def mop_up(self) -> int:
        """Return the mop-up bonus from White's view in pawnless endings."""
        counts = self.piece_counts
        white_pieces = (counts[WHITE_KNIGHT] + counts[WHITE_BISHOP] + counts[WHITE_ROOK]
                        + counts[WHITE_QUEEN])
        black_pieces = (counts[BLACK_KNIGHT] + counts[BLACK_BISHOP] + counts[BLACK_ROOK]
                        + counts[BLACK_QUEEN])
        white_king = SQUARE_64_FROM_120[self.king_squares[WHITE_SIDE]]
        black_king = SQUARE_64_FROM_120[self.king_squares[BLACK_SIDE]]
        if black_pieces == 0 and white_pieces > 0:
            bishop_square = -1
            if counts[WHITE_BISHOP] == 1:
                bishop_square = self.find_piece(WHITE_BISHOP)
            return mop_up_bonus(black_king, white_king, counts[WHITE_KNIGHT], counts[WHITE_BISHOP],
                                counts[WHITE_ROOK], counts[WHITE_QUEEN], bishop_square)
        if white_pieces == 0 and black_pieces > 0:
            bishop_square = -1
            if counts[BLACK_BISHOP] == 1:
                bishop_square = self.find_piece(BLACK_BISHOP)
            return -mop_up_bonus(white_king, black_king, counts[BLACK_KNIGHT],
                                 counts[BLACK_BISHOP], counts[BLACK_ROOK], counts[BLACK_QUEEN],
                                 bishop_square)
        return 0

    def white_score(self) -> int:
        """Return the tapered evaluation from White's view, without tempo."""
        counts = self.piece_counts
        mg_score = self.mg_score
        eg_score = self.eg_score
        if counts[WHITE_BISHOP] >= 2:
            mg_score += BISHOP_PAIR_MG
            eg_score += BISHOP_PAIR_EG
        if counts[BLACK_BISHOP] >= 2:
            mg_score -= BISHOP_PAIR_MG
            eg_score -= BISHOP_PAIR_EG
        phase = self.phase
        if phase > FULL_PHASE:
            phase = FULL_PHASE
        score = (mg_score * phase + eg_score * (FULL_PHASE - phase)) // FULL_PHASE
        if counts[WHITE_PAWN] == 0 and counts[BLACK_PAWN] == 0:
            score += self.mop_up()
        return score

    def evaluate(self) -> int:
        """Return the static evaluation for the side to move, with tempo."""
        score = self.white_score()
        if self.side == WHITE_SIDE:
            return score + TEMPO_BONUS
        return TEMPO_BONUS - score


def score_python_chess_board(board: chess.Board) -> tuple:
    """Return the midgame sum, endgame sum and phase of a python-chess board."""
    mg_score = 0
    eg_score = 0
    phase = 0
    for square, piece in board.piece_map().items():
        code = piece_code_of(piece)
        mg_score += MG_64[code * 64 + square]
        eg_score += EG_64[code * 64 + square]
        phase += PHASE_OF_CODE[code]
    return mg_score, eg_score, phase


def python_chess_mop_up(board: chess.Board) -> int:
    """Return the mop-up bonus from White's view for a pawnless python-chess board."""
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    pieces = board.knights | board.bishops | board.rooks | board.queens
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if white_king is None or black_king is None:
        return 0
    if pieces & black == 0 and pieces & white:
        bishops = board.bishops & white
        bishop_square = chess.lsb(bishops) if bishops else -1
        return mop_up_bonus(black_king, white_king, chess.popcount(board.knights & white),
                            chess.popcount(bishops), chess.popcount(board.rooks & white),
                            chess.popcount(board.queens & white), bishop_square)
    if pieces & white == 0 and pieces & black:
        bishops = board.bishops & black
        bishop_square = chess.lsb(bishops) if bishops else -1
        return -mop_up_bonus(white_king, black_king, chess.popcount(board.knights & black),
                             chess.popcount(bishops), chess.popcount(board.rooks & black),
                             chess.popcount(board.queens & black), bishop_square)
    return 0


def finish_python_chess_score(board: chess.Board, mg_score: int, eg_score: int,
                              phase: int) -> int:
    """Add the bishop pair, taper and mop-up; returns White's view without tempo."""
    if chess.popcount(board.bishops & board.occupied_co[chess.WHITE]) >= 2:
        mg_score += BISHOP_PAIR_MG
        eg_score += BISHOP_PAIR_EG
    if chess.popcount(board.bishops & board.occupied_co[chess.BLACK]) >= 2:
        mg_score -= BISHOP_PAIR_MG
        eg_score -= BISHOP_PAIR_EG
    if phase > FULL_PHASE:
        phase = FULL_PHASE
    score = (mg_score * phase + eg_score * (FULL_PHASE - phase)) // FULL_PHASE
    if board.pawns == 0:
        score += python_chess_mop_up(board)
    return score


def terminal_score(board: chess.Board):
    """Return the finished-game score from White's view, or None if play goes on."""
    has_move = False
    for move in board.generate_legal_moves():
        has_move = True
        break
    if not has_move:
        if board.is_check():
            if board.turn == chess.WHITE:
                return -TERMINAL_MATE_SCORE
            return TERMINAL_MATE_SCORE
        return 0
    if board.is_insufficient_material() or board.halfmove_clock >= 100:
        return 0
    if board.is_repetition(3):
        return 0
    return None


def evaluate_board(board: chess.Board) -> int:
    """Return the heuristic score of a python-chess board from White's view."""
    finished = terminal_score(board)
    if finished is not None:
        return finished
    mg_score, eg_score, phase = score_python_chess_board(board)
    return finish_python_chess_score(board, mg_score, eg_score, phase)


def python_chess_insufficient(board: chess.Board) -> bool:
    """Return True when neither side has mating material."""
    if board.pawns or board.rooks or board.queens:
        return False
    minors = board.knights | board.bishops
    white_minors = chess.popcount(minors & board.occupied_co[chess.WHITE])
    black_minors = chess.popcount(minors & board.occupied_co[chess.BLACK])
    return white_minors <= 1 and black_minors <= 1


def python_chess_move_delta(board: chess.Board, move: chess.Move) -> tuple:
    """Return the change in midgame, endgame and phase that a move causes."""
    from_square = move.from_square
    to_square = move.to_square
    piece_type = board.piece_type_at(from_square)
    if board.turn == chess.WHITE:
        own_offset = 0
        enemy_offset = BLACK_OFFSET
    else:
        own_offset = BLACK_OFFSET
        enemy_offset = 0
    code = piece_type + own_offset
    index = code * 64 + from_square
    mg_delta = -MG_64[index]
    eg_delta = -EG_64[index]
    phase_delta = 0
    placed = code
    if move.promotion:
        placed = move.promotion + own_offset
        phase_delta += PHASE_OF_CODE[placed]
    index = placed * 64 + to_square
    mg_delta += MG_64[index]
    eg_delta += EG_64[index]
    captured_type = board.piece_type_at(to_square)
    if captured_type:
        captured = captured_type + enemy_offset
        index = captured * 64 + to_square
        mg_delta -= MG_64[index]
        eg_delta -= EG_64[index]
        phase_delta -= PHASE_OF_CODE[captured]
    elif piece_type == PAWN and to_square == board.ep_square:
        if own_offset == 0:
            victim_square = to_square - 8
        else:
            victim_square = to_square + 8
        index = (PAWN + enemy_offset) * 64 + victim_square
        mg_delta -= MG_64[index]
        eg_delta -= EG_64[index]
    elif piece_type == KING and abs(to_square - from_square) == 2:
        if to_square > from_square:
            rook_from = from_square + 3
            rook_to = from_square + 1
        else:
            rook_from = from_square - 4
            rook_to = from_square - 1
        rook = ROOK + own_offset
        mg_delta += MG_64[rook * 64 + rook_to] - MG_64[rook * 64 + rook_from]
        eg_delta += EG_64[rook * 64 + rook_to] - EG_64[rook * 64 + rook_from]
    return mg_delta, eg_delta, phase_delta


def fen_position_key(board: chess.Board):
    """Return the first four FEN fields as a position key."""
    return book_key(board)


def detect_transposition_key() -> bool:
    """Return True if python-chess offers its private transposition key."""
    try:
        board = chess.Board()
        first = board._transposition_key()
        board.push_uci("g1f3")
        board.push_uci("g8f6")
        board.push_uci("f3g1")
        board.push_uci("f6g8")
        return first == board._transposition_key()
    except Exception as error:
        debug_print(f"Falling back to FEN keys: {error}")
        return False


USE_TRANSPOSITION_KEY = detect_transposition_key()


def position_key(board: chess.Board):
    """Return a hashable key identifying the position for Core A."""
    if USE_TRANSPOSITION_KEY:
        return board._transposition_key()
    return fen_position_key(board)


def encode_chess_move(move: chess.Move) -> int:
    """Pack a python-chess move into a small integer."""
    promotion = move.promotion or 0
    return move.from_square | (move.to_square << 6) | (promotion << 12)


def decode_chess_move(code: int) -> chess.Move:
    """Unpack an integer made by encode_chess_move."""
    promotion = (code >> 12) & 7
    if promotion == 0:
        return chess.Move(code & 63, (code >> 6) & 63)
    return chess.Move(code & 63, (code >> 6) & 63, promotion)


class PythonChessEngine:
    """Core A search: alpha-beta PVS over python-chess boards, correct by construction."""

    def __init__(self) -> None:
        """Create empty tables for one game."""
        self.tt = {}
        self.killers = new_killer_table()
        self.history = [0] * (2 * 64 * 64)
        self.nodes = 0
        self.hard_deadline = 0.0
        self.contempt = 0
        self.age = 0
        self.board = chess.Board()
        self.key_history = []
        self.iteration_move = None
        self.completed_best_uci = None
        self.last_depth = 0

    def prepare_search(self) -> None:
        """Age the tables once per move."""
        self.age += 1
        self.nodes = 0
        self.killers = new_killer_table()
        halve_history(self.history)
        self.tt = trim_tt(self.tt)
        self.completed_best_uci = None

    def draw_score(self, ply: int) -> int:
        """Return the draw score for the side to move at this ply."""
        if ply & 1:
            return self.contempt
        return -self.contempt

    def static_eval(self, mg_score: int, eg_score: int, phase: int) -> int:
        """Return the static evaluation for the side to move, with tempo."""
        board = self.board
        score = finish_python_chess_score(board, mg_score, eg_score, phase)
        if board.turn == chess.WHITE:
            return score + TEMPO_BONUS
        return TEMPO_BONUS - score

    def is_repetition(self, halfmove: int) -> bool:
        """Return True if the current key appeared since the last irreversible move."""
        keys = self.key_history
        current = keys[-1]
        index = len(keys) - 5
        stop = len(keys) - 1 - halfmove
        if stop < 0:
            stop = 0
        while index >= stop:
            if keys[index] == current:
                return True
            index -= 2
        return False

    def order_moves(self, moves: list, tt_code: int, ply: int) -> list:
        """Return the moves as packed sort keys, best first."""
        board = self.board
        killers = self.killers[ply]
        history = self.history
        history_base = 0 if board.turn == chess.WHITE else 4096
        enemy_pawns = board.pawns & board.occupied_co[not board.turn]
        pawn_attacks = chess.BB_PAWN_ATTACKS[board.turn]
        ep_square = board.ep_square
        keyed = []
        for move in moves:
            from_square = move.from_square
            to_square = move.to_square
            code = encode_chess_move(move)
            if code == tt_code:
                score = TT_MOVE_SCORE
            elif move.promotion:
                if move.promotion == chess.QUEEN:
                    victim = board.piece_type_at(to_square) or 0
                    score = QUEEN_PROMOTION_SCORE + SEE_VALUES[victim]
                else:
                    score = 0
            else:
                victim = board.piece_type_at(to_square)
                attacker = board.piece_type_at(from_square)
                if victim is None and attacker == PAWN and to_square == ep_square:
                    victim = PAWN
                if victim:
                    mvv_lva = 10 * victim - attacker
                    defended = pawn_attacks[to_square] & enemy_pawns
                    if SEE_VALUES[attacker] > SEE_VALUES[victim] and defended:
                        score = LOSING_CAPTURE_SCORE + mvv_lva
                    else:
                        score = GOOD_CAPTURE_SCORE + mvv_lva
                elif code == killers[0]:
                    score = FIRST_KILLER_SCORE
                elif code == killers[1]:
                    score = SECOND_KILLER_SCORE
                else:
                    score = history[history_base + from_square * 64 + to_square]
            keyed.append((score << 16) | code)
        keyed.sort(reverse=True)
        return keyed

    def record_quiet_cutoff(self, code: int, ply: int, depth: int) -> None:
        """Update killers and history after a quiet move caused a cutoff."""
        killers = self.killers[ply]
        if killers[0] != code:
            killers[1] = killers[0]
            killers[0] = code
        history_base = 0 if self.board.turn == chess.WHITE else 4096
        index = history_base + (code & 63) * 64 + ((code >> 6) & 63)
        self.history[index] += depth * depth
        if self.history[index] > HISTORY_MAX:
            halve_history(self.history)

    def think(self, game_board: chess.Board, game_keys: list, soft_limit: float,
              hard_limit: float, depth_cap: int, start_time: float) -> tuple:
        """Iteratively deepen on a copy of the game board; return (move, score, depth)."""
        self.prepare_search()
        self.hard_deadline = start_time + hard_limit
        # A copy keeps the game board intact if a timeout interrupts a push.
        board = game_board.copy()
        self.board = board
        self.key_history = list(game_keys)
        mg_score, eg_score, phase = score_python_chess_board(board)
        white_static = finish_python_chess_score(board, mg_score, eg_score, phase)
        our_static = white_static if board.turn == chess.WHITE else -white_static
        self.contempt = compute_contempt(phase, our_static)
        root_moves = list(board.generate_legal_moves())
        if not root_moves:
            return None, 0, 0
        tt_code = -1
        entry = self.tt.get(self.key_history[-1])
        if entry is not None:
            tt_code = entry & MOVE_MASK
        ordered = []
        for packed in self.order_moves(root_moves, tt_code, 0):
            ordered.append(decode_chess_move(packed & 0xFFFF))
        best_move = ordered[0]
        best_score = 0
        completed_depth = 0
        previous_score = None
        score_dropped = False
        for depth in range(1, depth_cap + 1):
            elapsed = time.perf_counter() - start_time
            limit = hard_limit if score_dropped else soft_limit
            if depth > 1 and elapsed > ITERATION_START_FRACTION * limit:
                break
            self.iteration_move = None
            try:
                score, move = self.search_root_moves(ordered, depth, mg_score, eg_score, phase)
            except SearchTimeout:
                if self.iteration_move is not None:
                    best_move = self.iteration_move
                break
            score_dropped = (previous_score is not None
                             and score <= previous_score - SCORE_DROP_MARGIN)
            previous_score = score
            best_score = score
            best_move = move
            completed_depth = depth
            self.completed_best_uci = move.uci()
            ordered = [best_move]
            best_code = encode_chess_move(best_move)
            for packed in self.order_moves(root_moves, best_code, 0):
                if (packed & 0xFFFF) != best_code:
                    ordered.append(decode_chess_move(packed & 0xFFFF))
            if abs(score) >= MATE_BOUND and MATE_VALUE - abs(score) <= depth:
                break
        self.last_depth = completed_depth
        return best_move, best_score, completed_depth

    def search_root_moves(self, ordered: list, depth: int, mg_score: int, eg_score: int,
                          phase: int) -> tuple:
        """Search every root move with PVS; return (best score, best move)."""
        board = self.board
        keys = self.key_history
        alpha = -INFINITY_SCORE
        beta = INFINITY_SCORE
        best_score = -INFINITY_SCORE
        best_move = ordered[0]
        first = True
        for move in ordered:
            mg_delta, eg_delta, phase_delta = python_chess_move_delta(board, move)
            board.push(move)
            keys.append(position_key(board))
            gives_check = board.is_check()
            child_mg = mg_score + mg_delta
            child_eg = eg_score + eg_delta
            child_phase = phase + phase_delta
            if first:
                score = -self.search_node(depth - 1, -beta, -alpha, 1, gives_check,
                                          child_mg, child_eg, child_phase)
                first = False
            else:
                score = -self.search_node(depth - 1, -alpha - 1, -alpha, 1, gives_check,
                                          child_mg, child_eg, child_phase)
                if alpha < score < beta:
                    score = -self.search_node(depth - 1, -beta, -alpha, 1, gives_check,
                                              child_mg, child_eg, child_phase)
            keys.pop()
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
                self.iteration_move = move
        return best_score, best_move

    def search_node(self, depth: int, alpha: int, beta: int, ply: int, in_check: bool,
                    mg_score: int, eg_score: int, phase: int) -> int:
        """Alpha-beta PVS node; returns a fail-soft score for the side to move."""
        self.nodes += 1
        if (self.nodes & TIME_CHECK_MASK) == 0 and time.perf_counter() > self.hard_deadline:
            raise SearchTimeout()
        board = self.board
        halfmove = board.halfmove_clock
        if halfmove >= 100 or self.is_repetition(halfmove) or python_chess_insufficient(board):
            return self.draw_score(ply)
        if ply >= MAX_PLY:
            return self.static_eval(mg_score, eg_score, phase)
        mate_alpha = ply - MATE_VALUE
        if alpha < mate_alpha:
            alpha = mate_alpha
        mate_beta = MATE_VALUE - ply - 1
        if beta > mate_beta:
            beta = mate_beta
        if alpha >= beta:
            return alpha
        if in_check:
            depth += 1
        if depth <= 0:
            return self.quiescence(alpha, beta, ply, in_check, mg_score, eg_score, phase)
        pv_node = beta - alpha > 1
        keys = self.key_history
        key = keys[-1]
        tt_code = -1
        entry = self.tt.get(key)
        if entry is not None:
            tt_code = entry & MOVE_MASK
            if not pv_node:
                tt_score = probe_tt_score(entry, depth, alpha, beta, ply)
                if tt_score is not None:
                    return tt_score
        moves = list(board.generate_legal_moves())
        if not moves:
            if in_check:
                return ply - MATE_VALUE
            return self.draw_score(ply)
        original_alpha = alpha
        best_score = -INFINITY_SCORE
        best_code = 0
        first = True
        for packed in self.order_moves(moves, tt_code, ply):
            code = packed & 0xFFFF
            move = decode_chess_move(code)
            quiet = move.promotion is None and not board.is_capture(move)
            mg_delta, eg_delta, phase_delta = python_chess_move_delta(board, move)
            board.push(move)
            keys.append(position_key(board))
            gives_check = board.is_check()
            child_mg = mg_score + mg_delta
            child_eg = eg_score + eg_delta
            child_phase = phase + phase_delta
            if first:
                score = -self.search_node(depth - 1, -beta, -alpha, ply + 1, gives_check,
                                          child_mg, child_eg, child_phase)
                first = False
            else:
                score = -self.search_node(depth - 1, -alpha - 1, -alpha, ply + 1, gives_check,
                                          child_mg, child_eg, child_phase)
                if alpha < score < beta:
                    score = -self.search_node(depth - 1, -beta, -alpha, ply + 1, gives_check,
                                              child_mg, child_eg, child_phase)
            keys.pop()
            board.pop()
            if score > best_score:
                best_score = score
                best_code = code
                if score > alpha:
                    alpha = score
                    if score >= beta:
                        if quiet:
                            self.record_quiet_cutoff(code, ply, depth)
                        break
        if best_score >= beta:
            flag = TT_LOWER
        elif best_score > original_alpha:
            flag = TT_EXACT
        else:
            flag = TT_UPPER
        store_tt_entry(self.tt, key, best_code, flag, depth, self.age, best_score, ply)
        return best_score

    def quiescence(self, alpha: int, beta: int, ply: int, in_check: bool, mg_score: int,
                   eg_score: int, phase: int) -> int:
        """Search captures and queen promotions until the position is quiet."""
        self.nodes += 1
        if (self.nodes & TIME_CHECK_MASK) == 0 and time.perf_counter() > self.hard_deadline:
            raise SearchTimeout()
        board = self.board
        if in_check:
            moves = list(board.generate_legal_moves())
            if not moves:
                return ply - MATE_VALUE
            best_score = -INFINITY_SCORE
            for packed in self.order_moves(moves, -1, ply):
                move = decode_chess_move(packed & 0xFFFF)
                mg_delta, eg_delta, phase_delta = python_chess_move_delta(board, move)
                board.push(move)
                score = -self.quiescence(-beta, -alpha, ply + 1, False, mg_score + mg_delta,
                                         eg_score + eg_delta, phase + phase_delta)
                board.pop()
                if score > best_score:
                    best_score = score
                    if score > alpha:
                        alpha = score
                        if score >= beta:
                            break
            return best_score
        stand_pat = self.static_eval(mg_score, eg_score, phase)
        if stand_pat >= beta or ply >= MAX_PLY:
            return stand_pat
        use_delta = phase > DELTA_ENDGAME_PHASE
        if use_delta and stand_pat + BIG_DELTA < alpha:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat
        best_score = stand_pat
        moves = []
        for move in board.generate_legal_captures():
            if move.promotion is None or move.promotion == chess.QUEEN:
                moves.append(move)
        own_pawns = board.pawns & board.occupied_co[board.turn]
        if board.turn == chess.WHITE:
            promoting_pawns = own_pawns & chess.BB_RANK_7
        else:
            promoting_pawns = own_pawns & chess.BB_RANK_2
        if promoting_pawns:
            empty_squares = ~board.occupied & chess.BB_ALL
            for move in board.generate_legal_moves(promoting_pawns, empty_squares):
                if move.promotion == chess.QUEEN:
                    moves.append(move)
        if not moves:
            return stand_pat
        for packed in self.order_moves(moves, -1, ply):
            move = decode_chess_move(packed & 0xFFFF)
            if use_delta and move.promotion is None:
                victim = board.piece_type_at(move.to_square) or PAWN
                if stand_pat + EG_VALUES[victim] + DELTA_MARGIN <= alpha:
                    continue
            mg_delta, eg_delta, phase_delta = python_chess_move_delta(board, move)
            board.push(move)
            score = -self.quiescence(-beta, -alpha, ply + 1, False, mg_score + mg_delta,
                                     eg_score + eg_delta, phase + phase_delta)
            board.pop()
            if score > best_score:
                best_score = score
                if score > alpha:
                    alpha = score
                    if score >= beta:
                        break
        return best_score


def compute_time_budget(remaining_s: float, ply: int, in_check: bool) -> tuple:
    """Return (soft limit, hard limit, depth cap) for this move."""
    usable = max(0.0, remaining_s - SAFETY_RESERVE_S - PER_MOVE_OVERHEAD_S)
    if remaining_s < PANIC_THRESHOLD_S:
        # Only spend the part above the instant threshold, so instant moves stay funded.
        spare = max(0.0, remaining_s - INSTANT_THRESHOLD_S)
        limit = min(PANIC_LIMIT_S, PANIC_FRACTION * spare)
        return limit, limit, PANIC_DEPTH_CAP
    our_moves_played = ply // 2
    moves_to_go = max(MIN_MOVES_TO_GO, EXPECTED_GAME_MOVES - our_moves_played // 2)
    moves_to_go = min(moves_to_go, max(1, (MAX_PLIES - ply + 1) // 2))
    soft = usable / moves_to_go
    if in_check:
        soft *= CHECK_TIME_FACTOR
    if ply < BOOK_MAX_PLY:
        soft *= OPENING_TIME_FACTOR
    hard = min(soft * HARD_MULTIPLIER, usable * HARD_FRACTION)
    soft = min(soft, hard)
    return soft, hard, MAX_SEARCH_DEPTH


def fen_fields_match(board: chess.Board, fen: str) -> bool:
    """Return True if placement, side, castling and en passant equal the FEN's."""
    ours = board.fen().split(" ")
    theirs = fen.split(" ")
    return ours[:4] == theirs[:4]


class GameState:
    """Everything act() remembers between the moves of one game."""

    def __init__(self) -> None:
        """Start a fresh game."""
        self.reset()

    def reset(self) -> None:
        """Forget the current game and return to the initial position."""
        self.known_history = []
        self.board = chess.Board()
        self.position_keys = [position_key(self.board)]
        self.book_active = True
        self.engine_a = PythonChessEngine()

    def load_from_fen(self, fen: str, history: list) -> None:
        """Trust the FEN alone when the history cannot be replayed."""
        debug_print("History does not match the FEN; loading the FEN directly.")
        self.board = chess.Board(fen)
        self.position_keys = [position_key(self.board)]
        self.known_history = list(history)

    def sync(self, observation: dict) -> None:
        """Bring the board up to date with the observation's history and FEN."""
        history = list(observation.get("history") or [])
        fen = observation.get("fen") or chess.STARTING_FEN
        known_count = len(self.known_history)
        if len(history) < known_count or history[:known_count] != self.known_history:
            self.reset()
        trusted = True
        for uci in history[len(self.known_history):]:
            try:
                self.board.push_uci(uci)
            except ValueError as error:
                debug_print(f"Cannot replay {uci}: {error}")
                trusted = False
                break
            self.position_keys.append(position_key(self.board))
            self.known_history.append(uci)
        if not trusted or not fen_fields_match(self.board, fen):
            self.load_from_fen(fen, history)
        side = "white" if self.board.turn == chess.WHITE else "black"
        if observation.get("color", side) != side:
            debug_print(f"FEN says {side} to move but we were told {observation.get('color')}")

    def book_move(self, ply: int, legal_moves: list):
        """Return a weighted-random book move, or None once the game leaves the book."""
        if not self.book_active or ply >= BOOK_MAX_PLY:
            return None
        try:
            entries = OPENING_BOOK.get(book_key(self.board))
            if entries:
                choice = pick_weighted(entries)
                if choice in legal_moves:
                    return choice
        except Exception as error:
            debug_print(f"Book lookup failed: {error}")
        self.book_active = False
        return None

    def instant_move(self, legal_moves: list) -> str:
        """Return the best capture with no search, for when the clock is nearly gone."""
        return best_capture(self.board, legal_moves)

    def search_move(self, observation: dict, start_time: float, remaining: float) -> str:
        """Run the time-managed Core A search."""
        ply = int(observation.get("ply", self.board.ply()))
        in_check = bool(observation.get("is_check", self.board.is_check()))
        soft_limit, hard_limit, depth_cap = compute_time_budget(remaining, ply, in_check)
        engine = self.engine_a
        move, score, depth = engine.think(self.board, self.position_keys, soft_limit,
                                          hard_limit, depth_cap, start_time)
        uci = move.uci() if move is not None else None
        core_name = "A"
        elapsed = time.perf_counter() - start_time
        nodes_per_second = int(engine.nodes / elapsed) if elapsed > 0 else 0
        debug_print(f"ply {ply} core {core_name} depth {depth} score {score} move {uci} "
                    f"nodes {engine.nodes} nps {nodes_per_second} time {elapsed:.3f}s "
                    f"soft {soft_limit:.3f}s hard {hard_limit:.3f}s")
        return uci


STATE = GameState()


def best_capture(board: chess.Board, legal_moves: list) -> str:
    """Return the highest MVV-LVA capture, or the first legal move if none exists."""
    best_uci = None
    best_score = -1
    try:
        for uci in legal_moves:
            move = chess.Move.from_uci(uci)
            if not board.is_capture(move):
                continue
            victim = board.piece_type_at(move.to_square) or PAWN
            attacker = board.piece_type_at(move.from_square) or PAWN
            score = 10 * victim - attacker
            if score > best_score:
                best_score = score
                best_uci = uci
    except Exception as error:
        debug_print(f"Capture fallback failed: {error}")
    if best_uci is not None:
        return best_uci
    return legal_moves[0]


def fallback_move(observation: dict) -> str:
    """Return a guaranteed legal move: last search result, best capture, or the first move."""
    legal_moves = list(observation.get("legal_moves") or [])
    if not legal_moves:
        return "0000"
    if STATE.engine_a.completed_best_uci in legal_moves:
        return STATE.engine_a.completed_best_uci
    try:
        board = chess.Board(observation["fen"])
        return best_capture(board, legal_moves)
    except Exception as error:
        debug_print(f"Fallback board failed: {error}")
    return legal_moves[0]


def choose_move(observation: dict) -> str:
    """Pick a move through the book, the clock rules and the search."""
    start_time = time.perf_counter()
    legal_moves = observation["legal_moves"]
    STATE.engine_a.completed_best_uci = None
    if len(legal_moves) == 1:
        return legal_moves[0]
    remaining = float(observation.get("time_remaining_s", DEFAULT_CLOCK_S))
    STATE.sync(observation)
    if remaining < INSTANT_THRESHOLD_S:
        return STATE.instant_move(legal_moves)
    ply = int(observation.get("ply", STATE.board.ply()))
    book_choice = STATE.book_move(ply, legal_moves)
    if book_choice is not None:
        return book_choice
    return STATE.search_move(observation, start_time, remaining)


def act(observation: dict) -> dict:
    """Return {"move": uci} for the current position; never raises and never resigns."""
    move = None
    try:
        move = choose_move(observation)
    except Exception as error:
        debug_print(f"act() recovered from {type(error).__name__}: {error}")
    try:
        if move not in observation["legal_moves"]:
            move = fallback_move(observation)
    except Exception as error:
        debug_print(f"Legality check failed: {error}")
        move = fallback_move(observation)
    return {"move": move}


def evaluate(fen: str) -> int:
    """Heuristic score from White's view: PeSTO tables, bishop pair and mop-up."""
    return evaluate_board(chess.Board(fen))


def minimax_node(board: chess.Board, depth: int, counter: list) -> int:
    """Plain minimax below one position, counting every visited position."""
    counter[0] += 1
    if depth == 0 or terminal_score(board) is not None:
        return evaluate_board(board)
    if board.turn == chess.WHITE:
        best_value = -math.inf
        for move in board.legal_moves:
            board.push(move)
            value = minimax_node(board, depth - 1, counter)
            board.pop()
            if value > best_value:
                best_value = value
    else:
        best_value = math.inf
        for move in board.legal_moves:
            board.push(move)
            value = minimax_node(board, depth - 1, counter)
            board.pop()
            if value < best_value:
                best_value = value
    return best_value


def alphabeta_node(board: chess.Board, depth: int, alpha: float, beta: float,
                   counter: list) -> int:
    """Minimax with alpha-beta pruning below one position, counting visited positions."""
    counter[0] += 1
    if depth == 0 or terminal_score(board) is not None:
        return evaluate_board(board)
    if board.turn == chess.WHITE:
        best_value = -math.inf
        for move in board.legal_moves:
            board.push(move)
            value = alphabeta_node(board, depth - 1, alpha, beta, counter)
            board.pop()
            if value > best_value:
                best_value = value
            if best_value > alpha:
                alpha = best_value
            if alpha >= beta:
                break
    else:
        best_value = math.inf
        for move in board.legal_moves:
            board.push(move)
            value = alphabeta_node(board, depth - 1, alpha, beta, counter)
            board.pop()
            if value < best_value:
                best_value = value
            if best_value < beta:
                beta = best_value
            if alpha >= beta:
                break
    return best_value


def minimax(fen: str, depth: int) -> dict:
    """Plain minimax to a fixed depth; returns {"value": score, "nodes": positions}."""
    counter = [0]
    value = minimax_node(chess.Board(fen), depth, counter)
    return {"value": value, "nodes": counter[0]}


def alphabeta(fen: str, depth: int) -> dict:
    """Alpha-beta search to a fixed depth; same value as minimax with fewer nodes."""
    counter = [0]
    value = alphabeta_node(chess.Board(fen), depth, -math.inf, math.inf, counter)
    return {"value": value, "nodes": counter[0]}

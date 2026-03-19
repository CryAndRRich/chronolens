from typing import List, Dict


class CONFIG_DATA:
    SEQ_LENGTH : int = 66
    FEATURE_COLS: List[str] = [f"feature_{i}" for i in range(1, SEQ_LENGTH + 1)]

    ATTRIBUTE_LIST: List[int] = [1, 2, 3, 4, 5, 6]
    ATTRIBUTE_COLS: List[str] = ["start_year", "attr_1", "attr_2", "attr_3", 
                                 "end_year", "attr_4", "attr_5", "attr_6"]

    EMBEDDING_DIM: int = 256
    BATCH_SIZE: int = 1024
    NUM_WORKERS: int = 2

    DAYS_IN_MONTH: List[int] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    DAYS_MAP: Dict[int, int] = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
    }
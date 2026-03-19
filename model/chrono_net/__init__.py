import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

__MODEL__ = {}


def register_model(name: str) -> object:
    """
    Decorator để đăng ký một mô hình mới vào hệ thống
    """
    def wrapper(cls):
        if __MODEL__.get(name, None) is not None:
            raise ValueError(f"Model {name} đã tồn tại")
        __MODEL__[name] = cls
        return cls
    return wrapper


def get_model(name: str, **kwargs) -> object:
    """
    Lấy một instance của mô hình đã đăng ký theo tên
    """
    if name not in __MODEL__:
        raise ValueError(f"Model {name} không tồn tại", available_models=list(__MODEL__.keys()))
    return __MODEL__[name](**kwargs)


import model.chrono_net.chrono_c
import model.chrono_net.chrono_g
import model.chrono_net.chrono_r
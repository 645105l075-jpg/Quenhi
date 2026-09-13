"""
data_generator.py
------------------
Sinh dữ liệu demo CHỈ trong phạm vi tuyến đường Lê Văn Việt và khu vực lân cận,
TP. Thủ Đức, TP.HCM (khu vực Quận 9 cũ).

Cấu trúc dữ liệu:
    node_id, latitude, longitude, waste_kg, service_time,
    time_window_start, time_window_end, is_depot, requires_small_vehicle
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pandas as pd


# ---------------------------------------------------------------------------
# Các tọa độ tham chiếu dọc tuyến Lê Văn Việt.
# Đây là tọa độ mô phỏng/xấp xỉ cho prototype, KHÔNG phải dữ liệu khảo sát
# thực địa chính xác từng mét.
# ---------------------------------------------------------------------------
LE_VAN_VIET_WAYPOINTS = [
    (10.8483, 106.7828),
    (10.8460, 106.7845),
    (10.8430, 106.7860),
    (10.8400, 106.7870),
    (10.8370, 106.7885),
    (10.8340, 106.7900),
    (10.8310, 106.7912),
    (10.8285, 106.7920),
]

# Depot / điểm tập kết giả định gần khu vực giữa tuyến.
DEPOT_LOCATION = (10.8400, 106.7870)

STUDY_AREA_LABEL = (
    "Khu vực thử nghiệm: Tuyến Lê Văn Việt – TP. Thủ Đức, TP.HCM"
)

# Độ lệch tối đa để mô phỏng các điểm nằm trên đường nhánh/hẻm
# kết nối với tuyến Lê Văn Việt.
_JITTER_LAT = 0.0020
_JITTER_LON = 0.0020


@dataclass
class DemoConfig:
    num_points: int = 25
    num_vehicles: int = 3
    seed: int = 42

    min_waste_kg: float = 80.0
    max_waste_kg: float = 450.0

    min_service_time_min: float = 3.0
    max_service_time_min: float = 12.0

    use_time_windows: bool = False
    tw_start_min: int = 0
    tw_end_min: int = 240
    tw_window_len_min: int = 90

    # Điểm lệch xa trục chính được gắn cờ là cần xe nhỏ.
    # Đây chỉ là giả định mô phỏng cho prototype.
    small_alley_jitter_ratio: float = 0.6


def generate_demo_data(
    config: DemoConfig | None = None,
) -> pd.DataFrame:
    """
    Sinh DataFrame gồm 1 depot + N điểm thu gom quanh tuyến Lê Văn Việt.

    Các điểm thu gom được tạo quanh các waypoint của tuyến chính,
    mô phỏng các đường nhánh/hẻm kết nối với Lê Văn Việt.
    """

    if config is None:
        config = DemoConfig()

    rng = random.Random(config.seed)

    # Giữ quy mô case study trong khoảng 20–30 điểm.
    n = max(20, min(config.num_points, 30))

    rows = [
        {
            "node_id": "DEPOT",
            "latitude": DEPOT_LOCATION[0],
            "longitude": DEPOT_LOCATION[1],
            "waste_kg": 0.0,
            "service_time": 0.0,
            "time_window_start": 0,
            "time_window_end": 24 * 60,
            "is_depot": True,
            "requires_small_vehicle": False,
        }
    ]

    for i in range(1, n + 1):
        base_lat, base_lon = rng.choice(LE_VAN_VIET_WAYPOINTS)

        jitter_lat = rng.uniform(-_JITTER_LAT, _JITTER_LAT)
        jitter_lon = rng.uniform(-_JITTER_LON, _JITTER_LON)

        lat = base_lat + jitter_lat
        lon = base_lon + jitter_lon

        waste = round(
            rng.uniform(
                config.min_waste_kg,
                config.max_waste_kg,
            ),
            1,
        )

        service_time = round(
            rng.uniform(
                config.min_service_time_min,
                config.max_service_time_min,
            ),
            1,
        )

        # Tính mức độ lệch khỏi trục Lê Văn Việt.
        jitter_ratio = (
            math.hypot(
                jitter_lat / _JITTER_LAT,
                jitter_lon / _JITTER_LON,
            )
            / math.sqrt(2)
        )

        requires_small_vehicle = (
            jitter_ratio > config.small_alley_jitter_ratio
        )

        if config.use_time_windows:
            latest_start = max(
                config.tw_start_min,
                config.tw_end_min - config.tw_window_len_min,
            )

            tw_start = rng.randint(
                config.tw_start_min,
                latest_start,
            )

            tw_end = min(
                config.tw_end_min,
                tw_start + config.tw_window_len_min,
            )
        else:
            tw_start = 0
            tw_end = 24 * 60

        rows.append(
            {
                "node_id": f"P{i:02d}",
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "waste_kg": waste,
                "service_time": service_time,
                "time_window_start": tw_start,
                "time_window_end": tw_end,
                "is_depot": False,
                "requires_small_vehicle": requires_small_vehicle,
            }
        )

    return pd.DataFrame(rows)


def load_points_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa dữ liệu người dùng upload về đúng schema.

    Cột bắt buộc:
        node_id, latitude, longitude, waste_kg

    Cột tùy chọn:
        service_time
        time_window_start
        time_window_end
        is_depot
        requires_small_vehicle

    Nếu không có depot, dòng đầu tiên sẽ được chọn làm depot.
    """

    required = {
        "node_id",
        "latitude",
        "longitude",
        "waste_kg",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Thiếu cột bắt buộc trong dữ liệu upload: "
            f"{sorted(missing)}"
        )

    out = df.copy()

    if "service_time" not in out.columns:
        out["service_time"] = 5.0

    if "time_window_start" not in out.columns:
        out["time_window_start"] = 0

    if "time_window_end" not in out.columns:
        out["time_window_end"] = 24 * 60

    if "is_depot" not in out.columns:
        out["is_depot"] = False
        out.loc[out.index[0], "is_depot"] = True

    if "requires_small_vehicle" not in out.columns:
        out["requires_small_vehicle"] = False

    # Nếu không có depot thì chọn dòng đầu tiên.
    if not out["is_depot"].any():
        out.loc[out.index[0], "is_depot"] = True

    # Đưa depot lên đầu DataFrame để các module khác có thể
    # sử dụng index 0 làm depot.
    depot_rows = out[out["is_depot"]]
    other_rows = out[~out["is_depot"]]

    out = pd.concat(
        [depot_rows, other_rows],
        ignore_index=True,
    )

    return out

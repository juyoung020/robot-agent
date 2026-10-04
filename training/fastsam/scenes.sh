# 장면 나누기: 평가 장면은 집(같은 건물의 다른 층·좌우 짝 포함) 단위로 뺀다. *_garden 은 *_int 와 같은 집이라 쓰지 않는다.
EVAL_SCENES=(house_double_floor_lower house_double_floor_upper Wainscott_0_int Wainscott_1_int office_cubicles_left
             office_cubicles_right restaurant_brunch grocery_store_cafe hotel_suite_small school_geography)
TRAIN_SCENES=(Beechwood_0_int Beechwood_1_int Benevolence_0_int Benevolence_1_int Benevolence_2_int Ihlen_0_int Ihlen_1_int
              Merom_0_int Merom_1_int Pomaria_0_int Pomaria_1_int Pomaria_2_int Rs_int gates_bedroom house_single_floor
              grocery_store_asian grocery_store_convenience grocery_store_half_stocked hall_arch_wood hall_conference_large
              hall_glass_ceiling hall_train_station hotel_gym_spa hotel_suite_large office_bike office_large
              office_vendor_machine restaurant_asian restaurant_cafeteria restaurant_diner restaurant_hotel restaurant_urban
              school_biology school_chemistry school_computer_lab_and_infirmary school_gym)

"""
This service is to get the filters and their values for a certain category available within the
data and its values. This is used to populate the filters in the UI.
"""

import os
import json

current_file = __file__
real_path = os.path.realpath(current_file)
filter_enums_dir = real_path.replace("enums.py", "")


class L4_Enums:
    def __init__(self):
        with open(filter_enums_dir + "l4_category_key_enums_master.json") as f:
            self.enums = json.load(f)

    def get_enum(self, enum_name: str):
        """
        Get the enum values for the given enum name.
        L4 enum names are like L2/L3/L4
        """
        enum_name_path = enum_name.split("/")
        return self.enums[enum_name_path[0]][enum_name_path[1]][enum_name_path[2]]


class L3_Enums:
    def __init__(self):
        with open(filter_enums_dir + "l3_category_key_enums_master.json") as f:
            self.enums = json.load(f)

    def get_enum(self, enum_name: str):
        """
        Get the enum values for the given enum name.
        L3 enum names are like L2/L3
        """
        enum_name_path = enum_name.split("/")
        return self.enums[enum_name_path[0]][enum_name_path[1]]


class L2_Enums:
    def __init__(self):
        with open(filter_enums_dir + "l2_category_key_enums_master.json") as f:
            self.enums = json.load(f)

    def get_enum(self, enum_name: str):
        """
        Get the enum values for the given enum name.
        L2 enum names are like L2
        """
        return self.enums[enum_name]

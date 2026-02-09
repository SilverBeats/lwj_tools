#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re

from src.lwj_tools.utils.tools import get_dir_file_path


def main():
    file_paths = get_dir_file_path(
        dir_path='old',
        skip_files=[re.compile(r'io*.*')],
        skip_dirs=['test'],
        should_skip_file=lambda s: 'nlg' in s,
    )

    print(file_paths)


if __name__ == '__main__':
    main()

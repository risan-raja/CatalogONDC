#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import subprocess

def main():
    try:
        while True:
            print("Starting program")
            p = subprocess.Popen(['python', 'query_gen.py'])
            p.wait()
            print("Program exited")
            time.sleep(5)
    except KeyboardInterrupt:
        return 

if __name__ == "__main__":
    main()

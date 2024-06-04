#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antoniosquicciarini
KAI ZEN
"""
import re

text = """
Data Sampling Rate: 256 Hz
*************************

Channels in EDF Files:
**********************
Channel 1: FP1-F7
Channel 2: F7-T7
Channel 3: T7-P7
Channel 4: P7-O1
Channel 5: FP1-F3
Channel 6: F3-C3
Channel 7: C3-P3
Channel 8: P3-O1
Channel 9: FP2-F4
Channel 10: F4-C4
Channel 11: C4-P4
Channel 12: P4-O2
Channel 13: FP2-F8
Channel 14: F8-T8
Channel 15: T8-P8
Channel 16: P8-O2
Channel 17: FZ-CZ
Channel 18: CZ-PZ
Channel 19: P7-T7
Channel 20: T7-FT9
Channel 21: FT9-FT10
Channel 22: FT10-T8
Channel 23: T8-P8

File Name: chb01_01.edf
File Start Time: 11:42:54
File End Time: 12:42:54
Number of Seizures in File: 0

File Name: chb01_02.edf
File Start Time: 12:42:57
File End Time: 13:42:57
Number of Seizures in File: 0

# ... (and so on)
"""
def EEG_metadata_extraction(text, args):
    # Extract data sampling rate
    sampling_rate_match = re.search(r'Data Sampling Rate:\s*(\d+)\s*Hz', text)
    sampling_rate = sampling_rate_match.group(1)

    print(f"Data Sampling Rate: {sampling_rate} Hz")
    args.sampling_rate = float(sampling_rate)

    # Extract channels
    channels_match = re.findall(r'Channel (\d+): (.+)', text)
    channels = {match[0]: match[1] for match in channels_match}
    print("Channels in EDF Files:")
    for channel_number, channel_name in channels.items():
        print(f"Channel {channel_number}: {channel_name}")
    args.channels = channels

    # Extract file information
    file_info_match = re.finditer(r'File Name: (.+?)\nFile Start Time: (\d+:\d+:\d+)\nFile End Time: (\d+:\d+:\d+)\nNumber of Seizures in File: (\d+)', text)

    for match in file_info_match:
        file_name, start_time, end_time, num_seizures = match.groups()
        print(f"File Name: {file_name}")
        print(f"File Start Time: {start_time}")
        print(f"File End Time: {end_time}")
        print(f"Number of Seizures in File: {num_seizures}")

    # If there are seizure times, extract them
    seizure_info_match = re.finditer(r'Seizure Start Time: (\d+) seconds\nSeizure End Time: (\d+) seconds', text)

    for match in seizure_info_match:
        start_time, end_time = match.groups()
        print(f"Seizure Start Time: {start_time} seconds")
        print(f"Seizure End Time: {end_time} seconds")

        
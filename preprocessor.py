import re
import pandas as pd
import numpy as np

def preprocess(data):
    # Matches common WhatsApp export timestamps (Android/iOS)
    # Examples:
    # 21/01/2020, 16:08 - ...
    # [21/01/20, 16:08:00] ...
    # 1/21/20, 4:08 PM - ...
    pattern = r'^\[?\d{1,2}[/-]\d{1,2}[/-]\d{2,4}[,\s]+\d{1,2}:\d{2}(?::\d{2})?\s*(?:[APap][.mM.]{1,4})?\]?(?:\s-\s|\s)'
    
    rows = []
    lines = data.strip().split('\n')
    current_date = None
    current_user = None
    current_message = []
    
    for line in lines:
        match = re.search(pattern, line)
        if match:
            # We found a new message. Save the previous one.
            if current_date is not None:
                rows.append([current_date, current_user, '\n'.join(current_message)])
                
            timestamp_str = match.group(0)
            msg_str = line[match.end():]
            
            # Clean up timestamp string to help pandas parse it
            ts_clean = re.sub(r'[\[\]\-]', '', timestamp_str).strip()
            if ts_clean.endswith(','): 
                ts_clean = ts_clean[:-1]
            
            current_date = ts_clean
            
            # Extract user and message
            user_msg_split = msg_str.split(': ', 1)
            if len(user_msg_split) == 2:
                current_user = user_msg_split[0]
                current_message = [user_msg_split[1]]
            else:
                current_user = "group_notification"
                current_message = [msg_str]
        else:
            # Multi-line message continuation
            current_message.append(line)
            
    # Append the last message
    if current_date is not None:
        rows.append([current_date, current_user, '\n'.join(current_message)])
        
    df = pd.DataFrame(rows, columns=['date', 'user', 'message'])
    
    # Parse dates efficiently handling multiple formats
    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    
    # Drop any rows where dates couldn't be parsed
    df = df.dropna(subset=['date'])
    
    # Add derived columns
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['only_date'] = df['date'].dt.date
    
    return df

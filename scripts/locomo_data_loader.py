import json

with open(r'data\correct_locomo10.json', 'r', encoding='utf-8') as f:
    data = json.load(f) # List
    # 一对speakers的数据的基本结构
    print(data[0].keys())
    # qa对的结构
    print(data[0]['qa'][0])
    # conversation的结构
    print(data[0]['conversation'].keys())
    print(data[0]['conversation']['session_1'][0])
    # event_summary的结构
    print(data[0]['event_summary'].keys())
    print(data[0]['event_summary']['events_session_1'])
    # observation的结构
    print(data[0]['observation'].keys())
    print(data[0]['observation']['session_1_observation'])
    # session_summary的结构
    print(data[0]['session_summary'].keys())
    print(data[0]['session_summary']['session_1_summary'])
    print(data[0]['sample_id'])
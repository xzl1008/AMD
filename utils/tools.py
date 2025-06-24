def save_record(path, content):
    try:
        with open(path, 'a', encoding='utf-8') as file:
            file.write(content)
    except Exception as e:
        print(f"Error: {e}")
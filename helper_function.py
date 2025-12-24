def get_raw_string(task_output):
    # 1. Check if the input is already a string
    if isinstance(task_output, str):
        return task_output
    
    # 2. Check if it's a CrewAI TaskOutput object (which has the .raw attribute)
    if hasattr(task_output, 'raw'):
        raw = task_output.raw
    else:
        raw = task_output

    # 3. Handle list or string conversion
    if isinstance(raw, list):
        return "\n".join([str(item) for item in raw])
    return str(raw)


def time_to_seconds(days = 0, hours = 0, minutes = 0, seconds = 5):
    """
    Converts argument's time to pure seconds
    """
    
    hours = hours + 24*days

    minutes = minutes + 60*hours

    seconds = seconds + 60*minutes

    return seconds


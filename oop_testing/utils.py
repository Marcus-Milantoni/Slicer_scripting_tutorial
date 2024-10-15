# Utility functions

def check_type(variable, expected_type, variable_name):
    if not isinstance(variable, expected_type):
        raise TypeError(f"The {variable_name} parameter must be a {expected_type.__name__}.")
    
def log_and_raise(logger, error_message, exception_type=Exception):
    logger.exception(error_message)
    raise exception_type(error_message)


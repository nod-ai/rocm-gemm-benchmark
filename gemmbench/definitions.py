from gemmbench import problems

def load_suite(top, suite):
    """Load suite from problems.py."""
    if suite is None:
        return list(problems.all())
    try:
        return list(getattr(problems, suite)())
    except:
        return []


def suite_description(top, suite):
    """Load suite from problems.py."""
    if suite is None:
        return None
    try:
        return getattr(problems, suite).__doc__
    except:
        return None


def load_configurations(top):
    """Load configurations from problems.py."""
    try:
        config = getattr(problems, 'configurations')
        return config
    except:
        return None

"""Utilities to scrape User Guide documentation."""


SKLEARN_USER_GUIDE_URL = "https://scikit-learn.org/dev/modules/"


def _user_guide_path_to_user_guide_url(path):
    """Convert a User Guide path to an User Guide URL.

    Parameters
    ----------
    path : :class:`pathlib.Path`
        The path to the User Guide documentation.

    Returns
    -------
    str
        The User Guide URL.
    """
    return SKLEARN_USER_GUIDE_URL + path.name

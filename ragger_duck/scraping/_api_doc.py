"""Utilities to scrape the API documentation."""

import importlib
import inspect
import re
import warnings

from numpydoc.docscrape import NumpyDocString
from sklearn.base import BaseEstimator, TransformerMixin

SKLEARN_API_URL = "https://scikit-learn.org/stable/modules/generated/"


def _api_path_to_api_url(path):
    """Convert a API path to an API URL.

    Parameters
    ----------
    path : :class:`pathlib.Path`
        The path to the API documentation.

    Returns
    -------
    str
        The API URL.
    """
    return SKLEARN_API_URL + path.name


def _extract_function_doc_numpydoc(function, import_name, html_source):
    """Extract documentation from a function using `numpydoc`.

    Parameters
    ----------
    function : callable
        The function to extract the documentation from.

    import_name : str
        The importh path of the function.

    html_source : str
        The link to the HTML source of the function.

    Returns
    -------
    list of dict
        List of dictionary with the keys `source` and `text` containing the
        documentation of the function.
    """
    try:
        docstring = NumpyDocString(function.__doc__)
    except TypeError as exc:
        # FIXME: it should be fixed upstream in scikit-learn
        warnings.warn(
            f"Fail to parse the docstring of {function.__name__}. Error message: {exc}"
        )
        return
    try:
        params = inspect.signature(function).parameters
    except ValueError as exc:
        warnings.warn(
            f"Fail to find the signature of {function.__name__}. Error message: {exc}"
        )
        return
    extracted_doc = []
    if len(params) > 0:
        chunk_doc = (
            f"{import_name}\n"
            + f"The parameters of {function.__name__} with their default "
            "values when known are:\n"
        )
        for param_name, param in params.items():
            if param.default is not param.empty:
                chunk_doc += f"- {param_name} (default={param.default})\n"
            else:
                chunk_doc += f"- {param_name}\n"
        extracted_doc.append({"source": html_source, "text": chunk_doc})
    # chunk about the summary of the class or function
    if docstring["Summary"] or docstring["Extended Summary"]:
        chunk_doc = (
            f"{import_name}\n"
            + f"The description of the {function.__name__} is as follow.\n"
        )
        if docstring["Summary"]:
            summary = "\n".join(docstring["Summary"])
            chunk_doc += f"{summary}\n"
        if docstring["Extended Summary"]:
            summary = "\n".join(docstring["Extended Summary"])
            chunk_doc += f"{summary}"
        extracted_doc.append({"source": html_source, "text": chunk_doc})
    # chunks about the parameters of the class or function
    if docstring["Parameters"]:
        for param_name, param_type, param_desc in docstring["Parameters"]:
            types = re.sub(" +", " ", param_type)
            desc = "\n".join(param_desc)
            chunk_doc = (
                f"{import_name}\n"
                + f"{param_name} is a parameter of the class "
                + f"{function.__name__}\n"
                + f"types: {types}\n"
                + f"description: {desc}"
            )
            extracted_doc.append({"source": html_source, "text": chunk_doc})
    # chunks about the attributes if we have a class
    if docstring["Attributes"]:
        for param_name, param_type, param_desc in docstring["Attributes"]:
            types = re.sub(" +", " ", param_type)
            desc = "\n".join(param_desc)
            chunk_doc = (
                f"{import_name}\n"
                + f"{param_name} is an attribute of the class "
                + f"{function.__name__}\n"
                + f"types: {types}\n"
                + f"description: {desc}"
            )
            extracted_doc.append({"source": html_source, "text": chunk_doc})
    # chunks about the returns or yields if we have a function
    if docstring["Returns"] or docstring["Yields"]:
        return_params = docstring["Returns"] or docstring["Yields"]
        for param_name, param_type, param_desc in return_params:
            types = re.sub(" +", " ", param_type)
            desc = "\n".join(param_desc)
            chunk_doc = (
                f"{import_name}\n"
                + f"{param_name} is returned by the function "
                + f"{function.__name__}\n"
                + f"types: {types}\n"
                + f"description: {desc}"
            )
            extracted_doc.append({"source": html_source, "text": chunk_doc})
    # chunks about the see also
    if docstring["See Also"]:
        chunk_doc = (
            f"{import_name}\n"
            + "The following functions or classes are related to "
            + f"{function.__name__}:\n"
        )
        for estimator, desc in docstring["See Also"]:
            chunk_doc += f"- {estimator[0][0]}"
            if desc:
                description = "\n".join(desc)
                chunk_doc += f": {description}\n"
            else:
                chunk_doc += "\n"
        extracted_doc.append({"source": html_source, "text": chunk_doc})
    # chunks about the notes
    if docstring["Notes"]:
        notes = "\n".join(docstring["Notes"])
        chunk_doc = f"{import_name}\n" + f"Notes: {notes}"
        extracted_doc.append({"source": html_source, "text": chunk_doc})
    # chunks about the examples
    if docstring["Examples"]:
        examples = "\n".join(docstring["Examples"])
        chunk_doc = (
            f"{import_name}\n"
            + f"Here is a usage example of {function.__name__}:\n{examples}"
        )
        extracted_doc.append({"source": html_source, "text": chunk_doc})

    # chunks about the references
    if docstring["References"]:
        references = "\n".join(docstring["References"])
        chunk_doc = (
            f"{import_name}\n"
            + f"Here are some references related to {function.__name__}:\n{references}"
        )
        extracted_doc.append({"source": html_source, "text": chunk_doc})

    return extracted_doc


class APINumPyDocExtractor(BaseEstimator, TransformerMixin):
    """Extract text from the API documentation using `numpydoc`.

    This function can process classes and functions. It extracts the information using
    `numpydoc` templates.
    """

    _parameter_constraints = {}

    def fit(self, X=None, y=None):
        """No-op operation, only validate parameters.

        Parameters
        ----------
        X : None
            This parameter is ignored.

        y : None
            This parameter is ignored.

        Returns
        -------
        self
            The fitted estimator.
        """
        self._validate_params()
        return self

    def transform(self, X):
        """Extract text from the API documentation.

        Parameters
        ----------
        X : :class:`pathlib.Path`
            The path to the API documentation folder.

        Returns
        -------
        output : list
            A list of dictionaries containing the source and text of the API
            documentation.
        """
        output = []
        for document in X.glob("*.html"):
            html_source = _api_path_to_api_url(document)
            full_name = document.stem
            try:
                module_name, class_or_function_name = full_name.rsplit(".", maxsplit=1)
            except ValueError as exc:
                # FIXME: specific scikit-learn hack
                if full_name == "dbscan-function":
                    module_name = "sklearn.cluster"
                    class_or_function_name = "dbscan"
                elif full_name == "fastica-function":
                    module_name = "sklearn.decomposition"
                    class_or_function_name = "fastica"
                elif full_name == "oas-function":
                    module_name = "sklearn.covariance"
                    class_or_function_name = "oas"
                else:
                    raise ValueError(
                        f"Fail to split the full name {full_name}. Error message: {exc}"
                    )
            if module_name == "sklearn.experimental":
                # FIXME: Only module are available in experimental
                continue
            elif "sklearn." in module_name:
                # FIXME: this is a hack to import the experimental modules
                # specifically for scikit-learn
                from sklearn.experimental import enable_halving_search_cv  # noqa
                from sklearn.experimental import enable_iterative_imputer  # noqa
            module = importlib.import_module(module_name)
            if not hasattr(module, class_or_function_name):
                warnings.warn(
                    f"Fail to find the class or function {class_or_function_name}. "
                    "It could be a module. Skipping it."
                )
                continue
            class_or_function = getattr(module, class_or_function_name)
            # get the documentation for the class or function
            extracted_doc = _extract_function_doc_numpydoc(
                class_or_function, full_name, html_source
            )
            if extracted_doc is None:
                continue
            output += extracted_doc

            # get the documentation for the class methods
            is_class = inspect.isclass(class_or_function)
            if is_class:
                for method_name in dir(class_or_function):
                    if method_name.startswith("_") or method_name.startswith("__"):
                        # private methods
                        continue
                    method = getattr(class_or_function, method_name)
                    if not inspect.isfunction(method):
                        continue
                    extracted_doc = _extract_function_doc_numpydoc(
                        method, f"{full_name}.{method_name}", html_source
                    )
                    if extracted_doc is not None:
                        output += extracted_doc
        return output

    def _more_tags(self):
        return {"X_types": ["string"], "stateless": True}

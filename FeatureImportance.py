'''
Available methods are the followings:
[1] FeatureImportance
[2] permutation_importance
[3] dfc_importanc
[4] drop_column_importance
[5] Axes2grid

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 30-11-2021

'''
import numpy as np, pandas as pd
import collections
from inspect import signature
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
from sklearn.base import clone
from sklearn.metrics import (roc_auc_score, 
                             accuracy_score,
                             f1_score, make_scorer, 
                             confusion_matrix)
import sklearn
from sklearn.ensemble import (RandomForestClassifier, 
                              ExtraTreesClassifier, 
                              RandomForestRegressor, 
                              ExtraTreesRegressor)
from sklearn.tree import (DecisionTreeRegressor, 
                          DecisionTreeClassifier, 
                          _tree)
from distutils.version import LooseVersion

plt.rcParams.update({'font.family':'sans-serif'})
plt.rcParams.update({'font.sans-serif':'Hiragino Sans GB'})
plt.rc('axes', unicode_minus=False)

__all__ = ["FeatureImportance",
           "permutation_importance", 
           "dfc_importance", 
           "drop_column_importance",
           "Axes2grid"]

if LooseVersion(sklearn.__version__) < LooseVersion("0.17"):
    raise Exception("TreeExplainer requires scikit-learn 0.17 or later")

def find_tree_paths(tree, node_id=0):
    
    '''
    Determine all paths through the tree as list of node-ids.
    
    Parameters
    ----------
    tree : sklearn.tree._tree.Tree object
        sklearn Tree 
    
    node_id : int, default=0

    Returns 
    -------
    paths : list of paths
        
    '''
    if node_id == _tree.TREE_LEAF:
        raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

    left_child  = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    if left_child != _tree.TREE_LEAF:
        left_paths  = find_tree_paths(tree, left_child)
        right_paths = find_tree_paths(tree, right_child)
        
        for path in left_paths: 
            path.append(node_id)
        for path in right_paths: 
            path.append(node_id)
            
        paths = left_paths + right_paths
    else: paths = [[node_id]]
        
    return paths

def _predict_tree(Tree, X):

    '''
    For a given estimator returns a tuple of [prediction, bias and 
    feature_contributions], such that prediction ≈ bias + 
    feature_contributions.
    
    Parameters
    ----------
    Tree : estimator object
        sklearn base estimator i.e. DecisionTreeRegressor or 
        DecisionTreeClassifier.
    
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples. The order of columns must match with one 
        that fits the model.
        
    Returns
    -------
    direct_prediction : np.ndarray
        The predicted class probabilities of an input sample `X`. The 
        shape is (n_samples,) for regression and (n_samples, n_classes) 
        for classification.

    biases : np.ndarray
        The estimator bias of an input sample. The shape is (n_samples,) 
        for regression and (n_samples, n_classes) for classification.

    contributions : np.ndarray
        Feature contribution of X. The shape is (n_samples, n_features) 
        for regression or (n_samples, n_features, n_classes) for 
        classification, denoting contribution from each feature.
        
    '''
    # Get leave nodes for all instances in X
    leaves = Tree.apply(np.array(X))
    
    # Get all paths from tree and order node.
    paths = find_tree_paths(Tree.tree_)
    for path in paths: path.reverse()

    # Map leaves to paths.
    leaf_to_path = dict((path[-1], path) for path in paths)
  
    # Remove the single-dimensional inner arrays.
    # values : number of sample wrt. class in each node.
    values = Tree.tree_.value.squeeze(axis=1)
    
    # Reshape if squeezed into a single float
    if len(values.shape) == 0:
        values = np.array([values])
        
    if isinstance(Tree, DecisionTreeRegressor):
        
        # We require the values to be the same shape as the biases
        values = values.squeeze(axis=1)
        biases = np.full(X.shape[0], values[paths[0][0]])
        line_shape = X.shape[1]
        
    elif isinstance(Tree, DecisionTreeClassifier):
        
        # scikit stores category counts, we turn them into probabilities
        normalizer = values.sum(axis=1, keepdims=True)
        values /= np.where(normalizer==0, 1, normalizer)

        # Create biases and contribution matrix (n_features)
        biases = np.tile(values[paths[0][0]], (X.shape[0], 1))
        line_shape = (X.shape[1], 2)
    
    # Predictions of X
    direct_prediction = values[leaves]
    
    # Convert into python list, accessing values will be faster
    values_list = list(values)
    feature_index = list(Tree.tree_.feature)
    unq_leaves  = np.unique(leaves)
    unq_contribs = {}

    for leaf in unq_leaves:
        path = leaf_to_path[leaf]
        contribs = np.zeros(line_shape)
        for i in range(len(path)-1):
            contrib = values_list[path[i+1]] - values_list[path[i]]
            contribs[feature_index[path[i]]] += contrib
        unq_contribs[leaf] = contribs

    # Feature contribution of X
    contributions = np.array([unq_contribs[leaf] for leaf in leaves])

    return direct_prediction, biases, contributions

def IterativeMean(i, current=None, new=None):
    
    '''
    Iteratively calculates mean using
    http://www.heikohoffmann.de/htmlthesis/node134.html
    
    Parameters
    ----------
    i : int 
        Non-negative integer, iteration.
    
    current : np.ndarray, default=None
        Current value of mean.
    
    new : np.ndarray, default=None
        New value to be added to mean.
    
    Returns
    -------
    numpy array, updated mean
    '''
    if current is None: return new
    else: return current + ((new - current) / (i + 1))

def _predict_forest(estimators, X):
    
    '''
    For given multiple estimators (forest) returns a tuple of 
    [prediction, bias and feature_contributions], such that 
    prediction ≈ bias + feature_contributions.
    
    Parameters
    ----------
    estimators : list of estimators
        sklearn tree-based estimator i.e. DecisionTreeRegressor or 
        DecisionTreeClassifier.
    
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples. The order of columns must match with one 
        that fits the model.
        
    Returns
    -------
    mean_pred : np.ndarray
        Averge of predicted class probabilities of an input sample. 
        The shape is (n_samples,) for regression and (n_samples, 
        n_classes) for classification.

    mean_bias : np.ndarray
        The averge of estimator biases. The shape is (n_samples,) for 
        regression and (n_samples, n_classes) for classification.

    mean_contribs : np.ndarray
        The averge of feature contributions. The shape is (n_samples, 
        n_features) for regression or (n_samples, n_features, n_classes) 
        for classification, denoting contribution from each feature.
         
    '''
    mean_pred = None
    mean_bias = None
    mean_contribs = None

    for i,Tree in enumerate(estimators):
        pred, bias, contribs = _predict_tree(Tree, X)
        
        # Update means.
        mean_pred = IterativeMean(i, mean_pred, pred)
        mean_bias = IterativeMean(i, mean_bias, bias)
        mean_contribs = IterativeMean(i, mean_contribs, contribs)

    return mean_pred, mean_bias, mean_contribs

class TreeExplainer_base:
    
    '''
    Determine Directional feature contribution of X given estimator.
    
    Parameters
    ----------
    estimator : estimator object
        The object must be the following scikit-learn estimator:
        - DecisionTreeRegressor
        - DecisionTreeClassifier
        - ExtraTreeRegressor
        - ExtraTreeClassifier
        - RandomForestRegressor
        - RandomForestClassifier

    '''
    def __init__(self, estimator):
        
         # Only single out response variable supported,
        if estimator.n_outputs_ > 1:
            raise ValueError("Multilabel classification trees not supported")
        
        classifiers = (DecisionTreeClassifier, RandomForestClassifier, 
                       ExtraTreesClassifier)
        regressors  = (DecisionTreeRegressor , RandomForestRegressor ,
                       ExtraTreesRegressor)
        
        if isinstance(estimator, classifiers + regressors): 
            self.estimator = estimator
        else: raise ValueError("Wrong model type. Base learner needs to "
                               "be either DecisionTreeClassifier or "
                               "DecisionTreeRegressor.")
        
        if isinstance(self.estimator, classifiers): self.classifier = True
        else: self.classifier = False
        
    def fit(self, X):
        
        '''
        Determine a tuple of (prediction, bias, feature_contributions), 
        such that prediction ≈ bias + feature_contributions.

        Parameters
        ----------
        X : array-like, of shape (n_samples, n_features)
            The input samples. The order of columns must match with one 
            that fits `estimator`.

        Attributes
        ----------
        prediction : np.ndarray
            The predicted class probabilities of an input sample `X`. The 
            shape is (n_samples,) for regression and (n_samples, n_classes) 
            for classification.

        bias : np.ndarray
            The estimator bias of an input sample `X`. The shape is 
            (n_samples,) for regression and (n_samples, n_classes) for 
            classification.
   
        contributions : np.ndarray
            Feature contribution of X. The shape is (n_samples, n_features) 
            for regression or (n_samples, n_features, n_classes) for 
            classification, denoting contribution from each feature.

        '''
        base_estimators = (DecisionTreeClassifier, DecisionTreeRegressor)
        if isinstance(self.estimator, base_estimators):
            ret_vals = _predict_tree(self.estimator, X)
        else: ret_vals = _predict_forest(self.estimator.estimators_, X)
        self.prediction, self.bias, self.contributions = ret_vals

        return self        

class FeatureImportance():
    
    '''
    This function determines the most predictive features through use 
    of "Feature importance", which is the most useful interpretation 
    tool. This function is built specifically for "scikit-learn" 
    RandomForestClassifier. 
    
    Parameters
    ----------
    methods : list of str, default=None
        If None, it defaults to all methods. The function to measure 
        the importance of variables. Supported methods are
        
            "gain" : average infomation gain (estimator.
                     feature_importances_).
            "dfc"  : directional feature constributions [3] 
            "perm" : permutation importance [4]
            "drop" : drop-column importance [5] 
                     
    scoring : list of functions, default=None
        List of sklearn.metrics functions for classification that 
        accept parameters as follows: `y_true`, and `y_pred` or 
        `y_score`. If None, it defaults to `f1_score`, 
        `accuracy_score`, and `roc_auc_score` [2].
        
    max_iter : int, default=10
        Maximum number of iterations of the algorithm for a single 
        predictor feature. This is relevant when "perm" is selected.

    random_state : int, default=None
        At every iteration, it controls the randomness of value 
        permutation of a single predictor feature. This is relevant 
        when "perm" is selected.

    References
    ----------
    .. [1] https://explained.ai/rf-importance/index.html
    .. [2] https://scikit-learn.org/stable/modules/model_evaluation.html
    .. [3] <class TreeExplainner_base>
    .. [4] <function permutation_importance>
    .. [5] <function drop_column_importance>
    
    Attributes
    ----------
    importances_ : collections.namedtuple
        A tuple subclasses with named fields as follows:
        - features      : list of features
        - gain_score    : Infomation gains
        - dfc_score     : Directional Feature Constributions
        - permute_score : Permutation importances
        - drop_score    : Drop-Column importances

    info : pd.DataFrame
        Result dataframe.

    result_ : Bunch
        Dictionary-like object, with the following attributes.
        - gain_score    : estimator.feature_importances_
        - dfc_score     : results from "dfc_importance"
        - permute_score : results from "permutation_importance"
        - drop_score    : results from "drop_column_importance"
    
    '''
    def __init__(self, methods=None, scoring=None, 
                 max_iter=10, random_state=0):
        
        if methods is None: 
            self.methods = {"gain", "dfc", "perm", "drop"}
        else: self.methods = methods
        
        self.scoring = scoring 
        self.max_iter = max_iter
        self.random_state = random_state
       
    def fit(self, estimator, X_train, y_train):

        '''
        Fit model
        
        Parameters
        ----------
        estimator : estimator object
            Fitted RandomForestClassifier estimator.
            
        X_train : pd.DataFrame, of shape (n_samples, n_features) 
            The training input samples. 

        y_train : array-like of shape (n_samples,)
            The training target labels (binary).
        
        Attributes
        ----------
        importances_ : collections.namedtuple
            A tuple subclasses with named fields as follows:
            - features      : list of features
            - gain_score    : Infomation gains
            - dfc_score     : Directional Feature Constributions
            - permute_score : Permutation importances
            - drop_score    : Drop-Column importances
    
        info : pd.DataFrame
            Result dataframe.
        
        result_ : Bunch
            Dictionary-like object, with the following attributes.
            - gain_score    : estimator.feature_importances_
            - dfc_score     : results from "dfc_importance"
            - permute_score : results from "permutation_importance"
            - drop_score    : results from "drop_column_importance"
                
        '''
        
        # Convert `X_train` to pd.DataFrame
        X = _to_DataFrame(X_train)
        info = dict(features = list(X_train))
        self.result_ = dict(features = list(X_train))
        mean_score = np.zeros(X.shape[1])
    
        # Infomation Gain (sklearn)
        if "gain" in self.methods:
            importances = estimator.feature_importances_
            self.result_.update({"gain_score" : importances})
            info.update({"gain_score" : importances})
            mean_score += importances
        
        # Directional Feature Constributions.
        if "dfc" in self.methods:
            result = dfc_importance(estimator, X)
            self.result_.update({"dfc_score" : result})
            info.update({"dfc_score" : result["importances_mean"]})
            mean_score += result["importances_mean"]
           
        # Permutation importance
        if "perm" in self.methods:
            kwargs = {"scoring"  : self.scoring, 
                      "max_iter" : self.max_iter, 
                      "random_state" : self.random_state}
            result = permutation_importance(estimator, X, y_train, **kwargs)
            self.result_.update({"permute_score" : result})
            info.update({"permute_score" : result["importances_mean"]})
            mean_score += result["importances_mean"]
            
        # Drop-Columns importance
        if "drop" in self.methods:
            result = drop_column_importance(estimator, X, y_train,
                                            scoring = self.scoring)
            self.result_.update({"drop_score" : result})
            info.update({"drop_score" : result["importances_score"]})
            mean_score += result["importances_score"]
        
        # Model attributes.
        info.update({"mean_score" : mean_score/sum(mean_score)})
        Results = collections.namedtuple('Results', info.keys())
        self.importances_ = Results(**info)
        self.info = (pd.DataFrame(info).set_index("features").
                     sort_values(by="mean_score", ascending=False))
        
        return self
    
    def plotting(self, column=None, sort_by=None, max_display=None, 
                 ax=None, colors=None, barh_kwds=None, char_length=20, 
                 tight_layout=True):
    
        '''
        Horizontal bar plot of feature importances.

        Parameters
        ----------
        column : str, default=None
            Column name in `info`. If None, it defaults to "mean_score".

        sort_by : str, default=None
            The column in `info` to sort feature contributions. If None, 
            no sorting is implemented.

        max_display : int, greater than 1, default=None
            Maximum number of variables to be displayed. If None, it
            uses one plus the maximum number of features, between mean 
            and median of importances.

        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, ax is created with 
            default figsize.

        colors : list of color-hex, default=None
            Number of color-hex must be greater than or equal to 2 i.e.
            ["Positive Importance", "Negative Importance"]. If None, it 
            uses default colors from Matplotlib.

        barh_kwds : keywords, default=None
            Keyword arguments to be passed to "ax.barh". If None, it uses 
            default settings.

        char_length : int, greater than 1, default=20
            Length of feature characters to be displayed.

        tight_layout : bool, default=True
            If True, it adjusts the padding between and around subplots 
            i.e. plt.tight_layout().

        References
        ----------
        .. [1] https://github.com/andosa/treeinterpreter
        .. [2] http://blog.datadive.net/interpreting-random-forests/

        Returns
        -------
        ax : Matplotlib axis object

        '''
        ax = barh_base(self.info, column=column, sort_by=sort_by, 
                       max_display=max_display, ax=ax, colors=colors, 
                       barh_kwds=barh_kwds, char_length=char_length, 
                       tight_layout=tight_layout)
        return ax

def barh_base(info, column=None, sort_by=None, max_display=None, 
              ax=None, colors=None, barh_kwds=None, char_length=20, 
              tight_layout=True):
    
    '''
    Horizontal bar plot of feature importances.
    
    Parameters
    ----------
    info : pd.DataFrame
        Result dataframe from `FeatureImportance`.
    
    column : str, default=None
        Column name in `info`. If None, it defaults to "mean_score".
     
    sort_by : str, default=None
        The column in `info` to sort feature contributions. If None, 
        no sorting is implemented.

    max_display : int, greater than 1, default=None
        Maximum number of variables to be displayed. If None, it
        uses one plus the maximum number of features, between mean 
        and median of importances.
       
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
       
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to 2 i.e.
        ["Positive Importance", "Negative Importance"]. If None, it 
        uses default colors from Matplotlib.

    barh_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.barh". If None, it uses 
        default settings.
        
    char_length : int, greater than 1, default=20
        Length of feature characters to be displayed.
        
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
        
    References
    ----------
    .. [1] https://github.com/andosa/treeinterpreter
    .. [2] http://blog.datadive.net/interpreting-random-forests/

    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    
    # ===============================================================
    params = {"gain_score"   : "Infomation gain",
              "dfc_score"    : "Feature Constribution", 
              "permute_score": "Permutation", 
              "drop_score"   : "Drop-Column",
              "mean_score"   : "Mean of Score(s)"}
    # ---------------------------------------------------------------
    if column is not None:
        if column not in list(info):
            raise ValueError(f"column must be in {list(info)}. "
                             f"Got {column} instead.")
    else: column = "mean_score"
    features, values = np.array(info.index), info[column].values
    # ---------------------------------------------------------------
    if sort_by is not None:
        if sort_by not in list(info):
            raise ValueError(f"column must be in {list(info)}"
                             f" or None. Got {sort_by} instead.")
        else: sort_index = np.argsort(info[sort_by])[::-1]
    else: sort_index = np.argsort(values)[::-1]
    features, values = features[sort_index], values[sort_index]
    mean, median = max(np.mean(values),0), max(np.median(values),0)
    n_mean, n_median = sum(values>=mean), sum(values>=median)
    # ---------------------------------------------------------------
    if max_display is None:
        max_display = max(n_mean, n_median) + 1
    elif not isinstance(max_display, int):
        raise ValueError(f"max_display must be integer. "
                         f"Got {type(max_display)} instead.")
    max_display = max(max_display, 1)
    if max_display < len(values):
        n_other = len(values) - max_display + 1
        values  = np.r_[values[:max_display-1],
                        values[max_display-1:].sum()]
        features= np.r_[features[:max_display-1],
                        ["{:,d} others".format(n_other)]]
    # ---------------------------------------------------------------
    new_features = []
    for n,f in enumerate(features, 1):
        if ((n==n_median) | (n==n_mean)) & (f!=features[-1]):
            new_features.append(f"({n}) " + f)
        else: new_features.append(f)
    features = new_features  
    num_features = len(features)
    # ===============================================================
    
    # ===============================================================
    # Positive and Negative contributions
    x = np.arange(len(values))
    pos_values = np.where(values< 0, np.nan, values)
    neg_values = np.where(values>=0, np.nan, values)
    # ---------------------------------------------------------------
    # Limit length of characters (features)
    length = int(max(char_length, 1))
    features = ([f[:length] + "..." if len(f) > length 
                 else f for f in features])
    if ax is None:
        ax = plt.subplots(figsize=(8, num_features*0.45 + 1.2))[1]
    # ---------------------------------------------------------------
    # Default colors
    colors = ([ax._get_lines.get_next_color() for n in range(2)] 
              if colors is None else colors)
    # ===============================================================
    
    # ===============================================================
    # Draw invisible bars just for sizing the axes
    ax.barh(x, pos_values, facecolor="none", edgecolor="none")
    ax.barh(x, neg_values, facecolor="none", edgecolor="none")
    # ---------------------------------------------------------------    
    kwds = {"height": 0.8} if barh_kwds is None else barh_kwds
    pos_text = {"inside" : dict(textcoords='offset points', 
                                va="center", ha="right", fontsize=13, 
                                xytext=(-3,0), color="white"), 
                "outside": dict(textcoords='offset points', 
                                va="center", ha="left", fontsize=13, 
                                xytext=(+3,0), color=colors[0])}
    neg_text = {"inside" : dict(textcoords='offset points', 
                                va="center", ha="left", fontsize=13, 
                                xytext=(+3,0), color="white"), 
                "outside": dict(textcoords='offset points', 
                                va="center", ha="right", fontsize=13, 
                                xytext=(-3,0), color=colors[1])}
    # ---------------------------------------------------------------
    r = plt.gcf().canvas.get_renderer()
    bbox = ax.get_window_extent(renderer=r)
    x_min, x_max = ax.get_xlim()
    x0, x1 = bbox.x0, bbox.x1
    # ---------------------------------------------------------------
    # Draw the positive bars
    kwds.update(dict(facecolor=colors[0]))
    pos_x1 = []
    for nx,ny in enumerate(pos_values):
        if ~np.isnan(ny):
            
            args = ("{:+.2%}".format(ny), (ny, nx))
            barh_obj  = ax.barh(nx, ny, **kwds).get_children()[0]
            text_obj  = ax.annotate(*args, **pos_text["inside"])
            barh_bbox = barh_obj.get_window_extent(renderer=r)
            text_bbox = text_obj.get_window_extent(renderer=r)
            
            # if the text overflows bar then draw it after the bar
            if text_bbox.width * 1.15 > barh_bbox.width: 
                text_obj.remove()
                text_obj  = ax.annotate(*args, **pos_text["outside"])
                text_bbox = text_obj.get_window_extent(renderer=r)
                
            pos_x1.append(max([barh_bbox.x1, text_bbox.x1]))
    # ---------------------------------------------------------------
    # Draw the negative bars
    kwds.update(dict(facecolor=colors[1]))
    neg_x0 = []
    for nx,ny in enumerate(neg_values):
        if ~np.isnan(ny):
      
            args = ("{:+.2%}".format(ny), (ny, nx))      
            barh_obj  = ax.barh(nx, ny, **kwds).get_children()[0]
            text_obj  = ax.annotate(*args, **neg_text["inside"])
            barh_bbox = barh_obj.get_window_extent(renderer=r)
            text_bbox = text_obj.get_window_extent(renderer=r)
         
            # if the text overflows bar then draw it after the bar
            if text_bbox.width * 1.15 > barh_bbox.width: 
                text_obj.remove()
                text_obj  = ax.annotate(*args, **neg_text["outside"])
                text_bbox = text_obj.get_window_extent(renderer=r)
                
            neg_x0.append(min([barh_bbox.x0, text_bbox.x0*0.6]))
    # ===============================================================

    # Set ax.set_yticklabels
    # ===============================================================
    t0 = r"$\bf{Mean}$ = "+"{:.2%} ({:d})".format(mean, n_mean)
    t1 = r"$\bf{Median}$ = "+"{:.2%} ({:d})".format(median, n_median)
    textstr = ', '.join((t0,t1))
    props = dict(boxstyle='square', facecolor='white', alpha=0)
    ax.text(0, 1.01, textstr, transform=ax.transAxes, fontsize=13,
            va='bottom', ha="left", bbox=props)
    # ---------------------------------------------------------------
    pos_x1 = (0 if len(pos_x1)==0 else max(pos_x1)) - x0 - 10
    neg_x0 = (0 if len(neg_x0)==0 else min(neg_x0)) - x0 + 10
    ratio = (x_max - x_min) / bbox.width
    if x_min < 0: x_min = min((neg_x0 * ratio) + x_min, x_min)
    if x_max > 0: x_max = max((pos_x1 * ratio) + x_min, x_max)
    ax.set_xlim(x_min, x_max)
    # ---------------------------------------------------------------
    ax.set_yticks(x)
    ax.set_yticklabels(features, fontsize=13)
    ax.tick_params(axis='x', labelsize=10.5)
    t = ticker.PercentFormatter(xmax=1, decimals=0)
    ax.xaxis.set_major_formatter(t)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(7))
    ax.set_xlabel(f"Importance Value ({params[column]})",fontsize=13)
    # ---------------------------------------------------------------
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis='y', length=0)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(-0.5, len(x)-0.5)
    ax.invert_yaxis()
    ax.yaxis.grid(True, color="grey", linewidth=0.2, linestyle="--")
    if tight_layout: plt.tight_layout()
    # ===============================================================
    
    return ax

def _to_DataFrame(X) -> pd.DataFrame:
    
    '''
    If `X` is not `pd.DataFrame`, column(s) will be automatically 
    created with "Unnamed" format.
    
    Parameters
    ----------
    X : array-like or pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
    
    '''
    if not (hasattr(X,'shape') or hasattr(X,'__array__')):
        raise TypeError(f'Data must be array-like. ' 
                        f'Got {type(X)} instead.')
    elif isinstance(X, pd.Series):
        return pd.DataFrame(X)
    elif not isinstance(X, pd.DataFrame):
        try:
            z = int(np.log(X.shape[1])/np.log(10)+1)
            columns = ['Unnamed_{}'.format(str(n).zfill(z)) 
                       for n in range(1,X.shape[1]+1)]
        except: columns = ['Unnamed']
        return pd.DataFrame(X, columns=columns)
    return X

def dfc_importance(estimator, X_train):
    
    '''
    Directional Feature Contributions (DFCs)
    
    Parameters
    ----------    
    estimator : estimator object
        Fitted scikit-learn RandomForestClassifier estimator.
    
    X_train : array-like of shape (n_samples, n_features) 
        The training input samples.
    
    References
    ----------
    .. [1] https://pypi.org/project/treeinterpreter/
    .. [2] Palczewska et al, https://arxiv.org/pdf/1312.1121.pdf
    .. [3] Interpreting Random Forests, http://blog.datadive.net/
           interpreting-random-forests/)
    
    Returns
    -------   
    result : Bunch
        Dictionary-like object, with the following attributes.

        importances_mean : ndarray, shape (n_features,)
            Mean of feature importance over n_samples.

        importances_std : ndarray, shape (n_features,)
            Standard deviation of feature importance over n_samples.

        contributions : ndarray, shape (n_samples, n_features)
            Raw Directional Feature Contribution scores.
            
    '''
    explain = TreeExplainer_base(estimator).fit(X_train)
    pred = explain.prediction
    bias = explain.bias
    cont = explain.contributions
    abs_cont = abs(cont[:,:,1])
    importances = abs_cont/abs_cont.sum(axis=1).reshape(-1,1)
    result = {"contributions" : cont[:,:,1], 
              "importances_mean" : np.mean(importances, axis=0),
              "importances_std" : np.std(importances, axis=0)}
    return result

def __CalScore__(y_true, y_pred, y_score, scoring):
    
    '''Private function to compute mean score'''
    score_ = []
    for scorer in scoring:
        if "y_pred" in signature(scorer).parameters:
            score_.append(scorer(y_true, y_pred))
        else: score_.append(scorer(y_true, y_score))
    return float(np.nanmean(score_))

def permutation_importance(estimator, X_train, y_train, 
                           scoring=None, max_iter=10, random_state=0):
    
    '''
    Record baseline metric(s) by passing a validation set through the 
    Random Forest (scikit-learn). Permute the column values of a 
    single predictor feature and then pass all test samples back 
    through the estimator and recompute the metrics. 
    
    The importance of that feature is the difference between the 
    baseline and the drop in overall accuracy caused by permuting the 
    column.
    
    Parameters
    ----------    
    estimator : estimator object
        Fitted scikit-learn RandomForestClassifier estimator.
    
    X_train : array-like of shape (n_samples, n_features) 
        The training input samples. 

    y_train : array-like of shape (n_samples,)
        The training target labels (binary).
     
    scoring : list of functions, default=None
        List of sklearn.metrics functions for classification that 
        accept parameters as follows: `y_true`, and `y_pred` or 
        `y_score`. If None, it defaults to [f1_score, accuracy_score, 
        roc_auc_score] [2].
        
    max_iter : int, default=10
        Maximum number of iterations of the algorithm for a single 
        predictor feature.

    random_state : int, default=0
        At every iteration, it controls the randomness of value 
        permutation of a single predictor feature. 

    References
    ----------
    .. [1] https://explained.ai/rf-importance/index.html
    .. [2] https://scikit-learn.org/stable/modules/model_evaluation.
           html
    
    Returns
    -------   
    result : Bunch
        Dictionary-like object, with the following attributes.

        importances_mean : ndarray, shape (n_features,)
            Mean of feature importance over max_iter.

        importances_std : ndarray, shape (n_features,)
            Standard deviation over max_iter.

        importances : ndarray, shape (n_features, max_iter)
            Raw permutation importance scores.
  
    '''
    # Convert input into ndarray
    y_true = np.array(y_train).copy()
    X = X_train.copy()
    
    # Default metrics for `scoring`
    if scoring is None: 
        scoring = [f1_score, 
                   roc_auc_score, 
                   accuracy_score]
    
    # Calculate the baseline score.
    bs_score = estimator.predict_proba(X)
    bs_pred  = np.argmax(bs_score, axis=1)
    args = (y_true, bs_pred, bs_score[:,1], scoring)
    baseline = __CalScore__(*args)
    
    # Initialize parameters.
    importances = []
    rand = np.random.RandomState(random_state)
    
    for n in np.arange(X.shape[1]):
        
        X_permute, m, mean_score = X.copy(), 0, []
        while m < max_iter:
            
            # Permute the column values (var) and then pass 
            # all permuted samples back through the model. 
            X_permute.iloc[:,n] = rand.permutation(X.iloc[:,n])
            y_score = estimator.predict_proba(X_permute)
            y_pred  = np.argmax(y_score, axis=1)
            
            # Recompute the accuracy and measure against
            # baseline accuracy.
            args = (y_true, y_pred, y_score[:,1], scoring)
            mean_score.append(baseline - __CalScore__(*args))
            m = m + 1
            
        # Calculate mean score
        importances.append(mean_score)
    
    # Permutation-importance scores
    importances = np.array(importances)
    raw = importances/abs(importances).sum(axis=0)
    result = {"importances" : importances, 
              "importances_mean" : np.mean(raw, axis=1),
              "importances_std" : np.std(raw, axis=1)}
    
    return result

def drop_column_importance(estimator, X_train, y_train, scoring=None):
    
    '''
    Calculate a baseline performance score. Then drop a column 
    entirely, retrain the model, and recompute the performance score. 
    The importance value of a feature is the difference between the 
    baseline and the score from the model missing that feature. 
    
    Parameters
    ----------    
    estimator : estimator object
        Fitted scikit-learn RandomForestClassifier estimator.
    
    X_train : array-like of shape (n_samples, n_features) 
        The training input samples. 

    y_train : array-like of shape (n_samples,)
        The training target labels (binary).
        
    scoring : list of functions, default=None
        List of sklearn.metrics functions for classification that 
        accept parameters as follows: `y_true`, and `y_pred` or 
        `y_score`. If None, it defaults to [f1_score, accuracy_score, 
        roc_auc_score] [2].
    
    References
    ----------
    .. [1] https://explained.ai/rf-importance/index.html
    .. [2] https://scikit-learn.org/stable/modules/model_evaluation.
           html
           
    Retruns
    -------
    result : Bunch
        Dictionary-like object, with the following attributes.

        importances_score : ndarray, shape (n_features,)
            Feature importance scores.

        drop_score : ndarray, shape (n_features,)
            Raw mean scores.
            
    '''
    # Convert input into ndarray
    y_true = np.array(y_train).copy()
    X = X_train.copy()
    
    # Default metrics for `scoring`
    if scoring is None: 
        scoring = [f1_score, 
                   roc_auc_score, 
                   accuracy_score]
    
    # Calculate the baseline score.
    bs_score = estimator.predict_proba(X)
    bs_pred  = np.argmax(bs_score, axis=1)
    args = (y_true, bs_pred, bs_score[:,1], scoring)
    baseline = __CalScore__(*args)
    
    # Initialize parameters.
    drop_score  = np.zeros(X.shape[1])
    estimator_  = clone(estimator)
    col_indices = np.arange(X.shape[1])
    
    for i in col_indices:
        # Drop column and retrain model.
        X_drop  = X.iloc[:,col_indices[col_indices!=i]].copy()
        y_score = estimator_.fit(X_drop, y_true).predict_proba(X_drop)
        y_pred  = np.argmax(y_score, axis=1)

        # Recompute the accuracy and measure against
        # baseline accuracy.
        args = (y_true, y_pred, y_score[:,1], scoring)
        drop_score[i] = baseline - __CalScore__(*args)
    
    # Calculate `importances_score`
    importances_score = drop_score/abs(drop_score).sum()
    result = {"drop_score" : drop_score, 
              "importances_score" : importances_score}
    
    return result

def Axes2grid(n_axes=4, n_cols=2, figsize=(6,4.3), 
              locs=None, spans=None):
    
    '''
    Create axes at specific location inside specified regular grid.
    
    Parameters
    ----------
    n_axes : int, default=4
        Number of axes required to fit inside grid.
        
    n_cols : int, default=2
        Number of grid columns in which to place axis. This will also 
        be used to calculate number of rows given number of axes 
        (`n_axes`).
    
    figsize : (float, float), default=(6,4.3)
        Width, height in inches for an axis.
    
    locs : list of (int, int), default=None
        locations to place each of axes within grid i.e. (row, column). 
        If None, locations are created, where placement starts from 
        left to right, and then top to bottom.

    spans : list of (int, int), default=None
        List of tuples for axis to span. First entry is number of rows 
        to span to the right while the second entry is number of 
        columns to span downwards. If None, every axis will default to
        (1,1).

    Returns
    -------
    fig : Matplotlib figure object
        The Figure instance.
    
    axes : list of Matplotlib axis object
        List of Matplotlib axes with length of `n_axes`.
    
    '''
    # Calculate number of rows needed.
    n_rows = np.ceil(n_axes/n_cols).astype(int)
    
    # Default values for `locs`, and `spans`.
    if locs is None: 
        locs = product(range(n_rows),range(n_cols))
    if spans is None: spans = list(((1,1),)*n_axes)

    # Set figure size
    width, height = figsize
    figsize=(n_cols*width, n_rows*height)
    fig = plt.figure(figsize=figsize)
    
    # Positional arguments for `subplot2grid`.
    args = [((n_rows,n_cols),) + (loc,) + span 
            for loc,span in zip(locs, spans)]
    return fig, [plt.subplot2grid(*arg) for arg in args]
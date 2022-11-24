#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 17:15:25 2021

@author: alfonso
"""

import pandas as pd
from scipy.sparse.linalg import eigs
import networkx as nx


def nodes_coordinates(G, compact, sort_by, n_outliers, normalize, margin, jittering, scale):
    """ Main method of this class. It computes exact node positions according to the inputed parameters. """
         
    outliers = []
    
    
    df, least_eigenvalue = _get_xcoord(G, compact)  
    
    df = _get_ycoord(G, df, sort_by)
    
    
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    
    
    if n_outliers:
        df, outliers = _remove_outliers(df, n_outliers)
    
    if margin:
        df = _margin(df, margin)
        
    if jittering:
        df = _jittering(df, jittering)
        
    if normalize:
        df = _normalize(df)
        
    if scale != 'linear':
        df = scale(df, scale)
        
    
    pos = {row.node: Point(row.x, row.y) for _, row in df.iterrows()}
    
    limits = _get_limits(df)
    rotation = _get_rotation(df)
    
    return pos, limits, rotation, least_eigenvalue, outliers


def _get_L_matrix(G):

    """
    Computes the signed Laplacian of the networkx Graph G.

    Required arguments:
    ----------
    *G*:
        A networkx connected, undirected signed Graph, with edges equals to either -1 or +1.
    
    
    Returns:
    ----------
    *L*:
        A np.array, the Laplacian matrix of graph G.  
        
    """
    
    from numpy import zeros
    
    try:
        assert nx.is_connected(G) and not nx.is_directed(G)
    except:
        raise Exception('Bad input: network G must be connected and undirected.')
        
        
    ## L = D - A
    
    # Computing A
    A = nx.adjacency_matrix(G, nodelist=G.nodes(), weight='weight')
    try:
        assert A.min() in [-1, 0] and A.max() in [0, 1]
    except:
        raise Exception('Bad input: edge weights must be +1 or -1.')
    
    # Computing D
    D = zeros(A.shape)
    for i in range(D.shape[0]):
        D[i,i] = sum([abs(x) for x in A[i, :]]).sum()
           
    L = D - A
    
    return L




def _get_xcoord(G, compact):
    
    """
    Finds the smallest eigenvalue and the related eigenvector.
    Computes each node's position after the computed eigenvector.
    
    Finds minimal and maximal position of nodes, and computes balancement between the two sides.

    Required arguments:
    ----------
    *G*:
        A networkx connected, undirected signed Graph, with edges equals to either -1 or +1.  
        
    
    Returns:
    ----------
    *df*:
        A dataframe with nodes' x coordinates.
        
    *least_eigenvalue*:
        A float. The least eigenvalue will be printed as a label, as additional information.
        
    """
    


    # L = D - A
    L = _get_L_matrix(G)
    
    # least_eigenvector
    least_eigenvalue, least_eigenvector = eigs(L, k=1, which='SM', return_eigenvectors=True)
    least_eigenvalue = round(least_eigenvalue[0], 4)
    
    df = pd.DataFrame({
            'node': list(G.nodes()),
            'x': [round(least_eigenvector[k][0], 5) if compact else least_eigenvector[k][0] for k in range(len(least_eigenvector))]
            })
    
    
    return df, least_eigenvalue


def _normalize(df):
    """ Normalize x-coordinates between [-1, 1]. """
    
    x = df['x'].abs().max()
    df['x'] = df['x'] / x
    
    return df
    
    
    
def _margin(df, margin):
    """ Puts a margin, a blank band around y = 0, sized after the `margin` parameter. """
        
    df['x'] = df['x'].map(lambda x: x + margin if x >= 0 else x - margin)
    
    return df
    
    
    
def _jittering(df, jittering):
    """ Jitters the x-coordinates of an amount equal to `jittering`. Suggested is 0.05. """
    
    from random import uniform
    dx = (df['x'].max() - df['x'].min()) * jittering
    df['x'] = df['x'].map(lambda x: uniform(x - dx, x + dx))
    
    return df


    
    
def _scale(df, scale):
    """ Displaces nodes on a different scale. Accepted values are 'linear' or 'log'. """
    
    if scale not in ['log']:
        raise Exception("ValueError: `scale` must be one of 'linear' or 'log'.")
    
    import numpy as np
    
    # Log scale
    if scale == 'log':
        mx = df['x'].abs().min()
        
        if mx == 0:
            mx = df['x'].abs().nlargest(2).iloc[-1] # get second minimum value
            df['x'] = df['x'].map(lambda x: x + mx if x >= 0 else x - mx) # translate each value of mx
        
        df['x'] = df['x'] * 1/mx
        df['x'] = df['x'].map(lambda x: np.log(x) if x >= 0 else - np.log(-x))
        
    return df
        

        
def _remove_outliers(df, n_outliers):
    """
    Finds the n points that are most far from the others on the x-axis.
    It searches the outliers on the extremes of the distribution, not in the center.
    """
    
    tmp = df.copy()
    tmp = tmp.sort_values('x').reset_index()
    del tmp['index']
    outliers = []
    
    for n in range(n_outliers):
        
        top = tmp.loc[1, 'x'] - tmp.loc[0, 'x']
        bottom = tmp.loc[tmp.index[-1], 'x'] - tmp.loc[tmp.index[-2], 'x']
        
        if top > bottom:
            outliers += [tmp.loc[0, 'node']]
            tmp = tmp.tail(len(tmp) - 1)
        else:
            outliers += [tmp.loc[tmp.index[-1], 'node']]
            tmp = tmp.head(len(tmp) - 1)
        
        tmp = tmp.reset_index()
        del tmp['index']
        
        
    return tmp, outliers


def _get_ycoord(G, df, sort_by):
    """ Get y-coordinates for each node. It may depend on the `sort_by` parameter. """
    
    
    if sort_by:
        df['kind'] = nx.get_node_attributes(G, sort_by).values()
    else:
        df['kind'] = 0
        
    
    order = pd.DataFrame(df['node'])
    
    
    df = df.sort_values(['x', 'kind'])
    df = df.reset_index()
    df['y'] = 1
    df['y'] = df.groupby('x')['y'].transform(pd.Series.cumsum)
    df['y'] = df['y'] - 1
    df = pd.merge(order, df, on = 'node')
    
    return df
        

def _get_limits(df):
    """ Computes limits of printable area. """
    
    minX = df['x'].min()
    maxX = df['x'].max()
    maxY = df['y'].max()
    
    if maxY == 0:
        maxY = 4
    
    return [minX, maxX, maxY]
    
    
def _get_rotation(df):
    """ Computes the rotation of the x-axis, depending on how many nodes belong to each side. """
    
    # Number of nodes in left and right side of the plot
    left = len(df.loc[df['x'] < 0])
    right = len(df.loc[df['x'] > 0])
                  
    # rotation is a function of how many nodes are in each side
    # rotation is bound to [-15, +15] degrees
    try:
        rot = (left / (left + right)) * 30 - 15
    except:
        rot = 0
        
    return rot
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:27:55 2021

@author: alfonso
"""
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.collections import PatchCollection
from functools import wraps
from collections import namedtuple



def _draw_signed_networkx_edges(G, 
                                ax, 
                                pos, 
                                limits, 
                                edge_alpha = 1, 
                                linewidth = .6, 
                                edges_linestyle = '-',
                                edges_color = None,
                                show_edges = None,
                                highlight_edges = None,
                                outliers = []):
    
    """
    Draws the edges of G.

    Required arguments:
    ----------
    *G*:
        A networkx connected, undirected signed Graph, with edges equals to either -1 or +1.
        
    *ax*:
        The ax to draw the edge on.
        
    *pos*:
        A dict, with the x and y coordinates for each node in G. An entry is shaped like { node_id : Point( x , y ) }
    
    *limits*:
        A 3-item tuple, respectively minX, maxX, maxY.
        
    *edge_aplha*:
        A float. Alpha of the edge to be drawn.
    
    *linewidth*:
        A float. Thickness of the edge to be drawn.
        
    *edge_linestyle*:
        A string. Default is '-', but it can be customized according to matplotlib linestyles.
    
    *edges_color*:
        Either a dict or a list. 
        If a dict, 'edges_color' must contain a color for each of the entries
            'frustrated_positive', 'frustrated_negative', 'balanced_positive', 'balanced_negative'.
        If a list, it must contain as many colors as many edges in G.
        
    *show_edges*:
        A string, one of "frustrated", "balanced", "frustrated_positive", "frustrated_negative", "balanced_positive", "balanced_negative",
        or "all". What kind of edges to show, while the other edges won't be drawed. 
        Default is "all". 'show_edges' and 'highlight_edges' are mutually exclusive.
        
    *highlight_edges*:
        A string, one of "frustrated", "balanced", "frustrated_positive", "frustrated_negative", "balanced_positive", "balanced_negative",
        or "all". What kind of edges to show regularly, while the other edges will be drawed in grey, with opacity = 0.01.
        Default is "all". 'show_edges' and 'highlight_edges' are mutually exclusive.
        
    *outliers*:
        A list of nodes id, outliers not to be shown. Do not print arcs that involve these outliers.
        
    """

              
    kinds = ['balanced_positive', 'balanced_negative', 'frustrated_positive', 'frustrated_negative']     
    edges = {key: [] for key in kinds}
    
    ## Divide each edge by its kind
    ## Keep them in a dict { kind: list of edges }
    for source, target, w in list(G.edges(data = True)):
    
        if (source in outliers) or (target in outliers):
            continue
        
        weight = w['weight']
    
        p1, p2 = [pos[source], pos[target]]
        
        
        # Check edge kind        
        if (np.sign(p1.x) != np.sign(p2.x)) and weight == 1:
            kind = 'frustrated_positive'
        elif (np.sign(p1.x) == np.sign(p2.x)) and weight == -1:
            kind = 'frustrated_negative'
        elif weight == 1:
            kind = 'balanced_positive'
        else:
            kind = 'balanced_negative'
        
        edges[kind] += [(p1, p2, weight)]
        
        
    ## Determine width, alpha and color of each kind of nodes 
    ## according to show_edges / highlight_edges
    for kind in kinds:
        
        # Don't show this kind of arc
        if (show_edges != 'all') and (show_edges not in kind):
            continue
        
        # This kind of arc is opaque
        if (highlight_edges != 'all') and (highlight_edges not in kind):
            alpha = 0.03
            lw = linewidth * .5
            color = 'lightgrey'
            zorder = 1
            
        # Regularly show this kind of arc
        else:
            alpha = edge_alpha
            color = edges_color[kind]
            lw = linewidth
            zorder = 2 if kind in ['frustrated_positive', 'balanced_negative'] else 3
        
        draw_edge_kind(ax, edges[kind], limits, lw, edges_linestyle, alpha, color, zorder)
    
           
        
def draw_edge_kind(ax,
              edges_kind,
              limits,
              linewidth,
              edge_linestyle,
              alpha,
              color,
              zorder):
    
    """
    Draws the edges of one kind.

    Required arguments:
    ----------
        
    *ax*:
        The ax to draw the edge on.
        
    *edges_kind*:
        A list of edges. An entry is shaped like (p1, p2, weight) where p1 and p2 are a Point( x , y )
        and weigth is either -1 or +1.
    
    *limits*:
        A 3-item tuple, respectively minX, maxX, maxY.
            
    *linewidth*:
        A float. Thickness of the edge to be drawn.
        
    *edge_linestyle*:
        A string. Default is '-', but it can be customized according to matplotlib linestyles.
        
    *alpha*:
        A float. Alpha of the edge to be drawn.
    
    *color*:
        A string. Color for this kind of edges.
        
    *zorder*:
        A numeric. z-axis order for this kind of edges (hidden edges must be put below visible edges).
        
    """
           
    patches = []
    
    for edge in edges_kind:
        
        p1, p2, weight = edge
        
        if p1.x == p2.x:
    
            # p1 must be the upper point, p2 must be the lower point
            if p1.y < p2.y:
                p1, p2 = p2, p1

            # E+i: same coordinates and friends, blue with horizontal-external bundling
            if weight == 1:
                
                if p1.x < 0:
                    patch = _vert_link(ax, p1, p2, orient = 'left', limits = limits)
                else:
                    patch = _vert_link(ax, p1, p2, orient = 'right', limits = limits)
                
                    

            # E-i: same coordinates but not friends, red with horizontal internal bundling
            elif weight == -1:
                if p1.x < 0:
                    patch = _vert_link(ax, p1, p2, orient = 'right', limits = limits)
                else:
                    patch = _vert_link(ax, p1, p2, orient = 'leeft', limits = limits)
                    

        else:

            # E+e: different coordinates and friends, blue with vertical-upper bundling
            if weight == 1:
                
                # p1 must be the left point, p2 must be the right point
                if p1.x < p2.x:
                    p1, p2 = p2, p1
                        
                        
                if p1.y == p2.y:
                    patch = _horiz_blue(ax, p1, p2, limits = limits)
                
                elif p1.y < p2.y:
                    patch = _diag_blue_asc(ax, p1, p2)
            
                else:
                    patch = _diag_blue_desc(ax, p1, p2)
                    
            

            # E-e: different coordinates but not friends, red with vertical-lower bundling
            elif weight == -1:

                # p1 must be the left point, p2 must be the right point
                if p1.x > p2.x:
                    p1, p2 = p2, p1
                
                patch = _red_link(ax, p1, p2, limits = limits)
                
                   
        patches.append(patch)
                
                
    
    patches = PatchCollection(patches, facecolor = 'none', linewidth = linewidth, edgecolor = color,  match_original=True, linestyle = edge_linestyle, alpha = alpha, zorder = zorder)
    ax.add_collection(patches)
    
    

def _horiz_blue(ax, p1, p2, limits) -> None:
    
    """
    Draws horizontal positive links.

    Required arguments:
    ----------
    *ax*:
        The ax to draw the edge on.
        
    *p1*:
        A Point. First of the two vertices between which the edge will be drawn.
    
    *p1*:
        A Point. First of the two vertices between which the edge will be drawn.
        
    *limits*:
        A 3-item tuple, respectively minX, maxX, maxY.
        
    *aplha*:
        A float. Alpha of the edge to be drawn.
    
    *linewidth*:
        A float. Thickness of the edge to be drawn.
        
    *linestyle*:
        A String. Default is '-' for continuous lines, see: Matplotlib linestyles.
    
    *color*:
        A string. Default is 'steelblue', but it can be customized. 
        
    
    Notes:
    ----------
    Points p1 and p2 are assumed to have:
        p1.y == p2.y
        p1.x < p2.x
        
    """
    
    
    Point = namedtuple('Point', ['x', 'y'])
    
    # Computing midpoints
    H = (p2.x - p1.x) / 2
    K = H + limits[2] / 10 # makes the horizontal humps more rounded.
    
    # Define Bezier curve
    mid1 = Point(p1.x, p1.y + K)
    mid2 = Point(p1.x + H, p1.y + K)
    mid3 = Point(p2.x, p2.y + K)
    Path = mpath.Path
    path_data = [
        (Path.MOVETO, p1),
        (Path.CURVE3, mid1),
        (Path.CURVE3, mid2),
        (Path.CURVE3, mid3),
        (Path.CURVE3, p2)
        ]
    
    # Add the path to ax
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path)
    
    return patch
    

def _vert_link(ax, p1, p2, limits, orient = 'left') -> None:
    
    """
    Draws vertical links, both negative (bound internally) and positive (bound externally).
    
    Required arguments:
    ----------
    *ax*:
        The ax to draw the edge on.
        
    *p1*:
        A Point. First of the two vertices between which the edge will be drawn.
    
    *p1*:
        A Point. First of the two vertices between which the edge will be drawn.
        
    *limits*:
        A 3-item tuple, respectively minX, maxX, maxY.
        
    *aplha*:
        A float. Alpha of the edge to be drawn.
    
    *linewidth*:
        A float. Thickness of the edge to be drawn.
    
    *linestyle*:
        A String. Default is '-' for continuous lines, see: Matplotlib linestyles.
    
    *color*:
        A string. Default is 'steelblue', but it can be customized. 
        
    *orient*:
        A string, can be either 'right' or 'left'. Red vertical edges should be bound internally, blue vertical edges externally.
        
    
    Notes:
    ----------
    Points p1 and p2 are assumed to have:
        p1.x == p2.x
        p1.y > p2.y
        
    """
    
    
    Point = namedtuple('Point', ['x', 'y'])
    
    # computing midpoints
    K = (p1.y - p2.y) / 2
#    H = (limits[1]/limits[2]) / 5 # makes vertical humps more rounded
    H = limits[1] / 20
    H *= min([(p1.y - p2.y) / limits[2], 1]) * 3 # makes a wider arc for nodes far apart
    
    
    if orient == 'right':
        H = -H
        
    mid1 = Point(p1.x - H, p1.y)
    mid2 = Point(p1.x - H, p1.y - K)
    mid3 = Point(p2.x - H, p2.y)
    
    # Define Bezier curve
    Path = mpath.Path
    path_data = [
        (Path.MOVETO, p1),
        (Path.CURVE3, mid1),
        (Path.CURVE3, mid2),
        (Path.CURVE3, mid3),
        (Path.CURVE3, p2)
        ]
    
    # add the Path to ax
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path)
    
    return patch
  

def _diag_blue_desc(ax, p1, p2) -> None:
    
    """
    Draws positive links where p1 is upper-left respect to p2.

    Required arguments:
    ----------
    *ax*:
        The ax to draw the edge on.
        
    *p1*:
        A Point. First of the two vertices between which the edge will be drawn.
    
    *p1*:
        A Point. First of the two vertices between which the edge will be drawn.
        
    *aplha*:
        A float. Alpha of the edge to be drawn.
    
    *linewidth*:
        A float. Thickness of the edge to be drawn.
    
    *linestyle*:
        A String. Default is '-' for continuous lines, see: Matplotlib linestyles.
        
    *color*:
        A string. Default is 'steelblue', but it can be customized. 
        
    
    Notes:
    ----------
    Points p1 and p2 are assumed to have:
        p1.y > p2.y
        p1.x < p2.x
        
    """
    Point = namedtuple('Point', ['x', 'y'])
    
    mid2 = Point(p2.x, p1.y)
    
    # Define Bezier curve
    Path = mpath.Path
    path_data = [
        (Path.MOVETO, p1),
        (Path.CURVE3, mid2),
        (Path.CURVE3, p2)
        ]
    
    # add the Path to ax
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path)
    
    return patch



def _diag_blue_asc(ax, p1, p2) -> None:
    """
    Draws positive links where p1 is lower-left respect to p2.

    Required arguments:
    ----------
    *ax*:
        The ax to draw the edge on.
        
    *p1*:
        A Point. First of the two vertices between which the edge will be drawn.
    
    *p1*:
        A Point. First of the two vertices between which the edge will be drawn.
        
    *aplha*:
        A float. Alpha of the edge to be drawn.
    
    *linewidth*:
        A float. Thickness of the edge to be drawn.
    
    *linestyle*:
        A String. Default is '-' for continuous lines, see: Matplotlib linestyles.
    
    *color*:
        A string. Default is 'steelblue', but it can be customized. 
        
    
    Notes:
    ----------
    Points p1 and p2 are assumed to have:
        p1.y < p2.y
        p1.x < p2.x
        
    """    
    Point = namedtuple('Point', ['x', 'y'])
    
    # Define Bezier curve
    mid2 = Point(p1.x, p2.y)
    Path = mpath.Path
    path_data = [
        (Path.MOVETO, p1),
        (Path.CURVE3, mid2),
        (Path.CURVE3, p2)
        ]
    
    # Add path to ax
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path)
    
    return patch



def _red_link(ax, p1, p2, limits) -> None:

    """
    Draws all kind of negative edges. Negative edges must be bundled below zero, regardelss of p1 and p2 position.

    Required arguments:
    ----------
    *ax*:
        The ax to draw the edge on.
        
    *p1*:
        A Point. First of the two vertices between which the edge will be drawn.
    
    *p1*:
        A Point. First of the two vertices between which the edge will be drawn.
        
    *limits*:
        A 3-item tuple, respectively minX, maxX, maxY.
        
        
    
    Notes:
    ----------
    Points p1 and p2 are assumed to have:
        p1.y != p2.y
        p1.x < p2.x
        
    """
    
    Point = namedtuple('Point', ['x', 'y'])
    
    # computing midpoints
    H = (p2.x - p1.x) / 2
    K = H + limits[2] / 8 # makes red edges wired and bundle, below 0
    
    # if points don't have same x
    if (p2.x - p1.x) != 0:
        mid1 = Point(p1.x, 0 - K)
        mid2 = Point(p1.x + H, 0 - K)
        mid3 = Point(p2.x, 0 - K)
    
    # if points have same x
    else:   
        min_eps = max([-limits[0], limits[1]]) / 20 # this defines min shift
        max_eps = max([-limits[0], limits[1]]) / 10 # this defines max shift
        eps = max(p1.y - p2.y, p2.y - p1.y) / limits[2] # vertical difference
        eps = eps + (max_eps - min_eps) * eps # min shif + proportional to vertical distance
        
        mid1 = Point(p1.x - eps, 0 - K)
        mid2 = Point(p1.x, 0 - K)
        mid3 = Point(p2.x + eps, 0 - K)
    
    # define the Bezier curve
    Path = mpath.Path
    path_data = [
        (Path.MOVETO, p1),
        (Path.CURVE3, mid1),
        (Path.CURVE3, mid2),
        (Path.CURVE3, mid3),
        (Path.CURVE3, p2)
        ]
    
    # add the Path to ax
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path)
    
    return patch

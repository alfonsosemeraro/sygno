"""

**********
Signed_networkx
**********

This package contains a data visualization tool for signed networks.
Given a connected and undirected signed network with edges weighted -1 or +1, 
this library gives a representation that highlights frustrated edges
and the overall balancement between two partitions of nodes.

This library is based on 

Galimberti E., Madeddu C., Bonchi F., Ruffo G. (2020) Visualizing Structural Balance in Signed Networks. 
In: Cherifi H., Gaito S., Mendes J., Moro E., Rocha L. (eds) Complex Networks and Their Applications VIII. 
COMPLEX NETWORKS 2019. Studies in Computational Intelligence, vol 882. Springer, Cham. 
https://doi.org/10.1007/978-3-030-36683-4_5

--------
repository available at https://www.github.com/alfonsosemeraro/pyplutchik
@author: Alfonso Semeraro <alfonso.semeraro@gmail.com>
"""

from collections import namedtuple
from scipy.sparse.linalg import eigsh
import networkx as nx
from numpy import zeros
import matplotlib.pyplot as plt
import numpy as np

from nodes_position import nodes_coordinates

#import line_profiler

__author__ = """Alfonso Semeraro (alfonso.semeraro@gmail.com)"""
__all__ = ['draw_signed_networkx',
           '_setup_axes1',
           '_draw_signed_networkx_nodes',
           '_draw_signed_networkx_edges',
           '_get_positions',
           '_get_L_matrix']




  


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
    
    if show_edges and highlight_edges:
        raise Exception("InputError: `show_edges` and `highlight_edges` must be mutually exclusive.")
        
    if not show_edges:
        show_edges = 'all'
        
    if not highlight_edges:
        highlight_edges = 'all'
    
    if show_edges and show_edges not in ['all', 'frustrated', 'balanced', 
                          'positive', 'negative',
                          'frustrated_positive', 'frustrated_negative',
                          'balanced_positive', 'balanced_negative']:
        raise Exception("InputError: show_edges must be one of 'all' (default), 'balanced', 'frustrated', 'positive', 'negative', "+
                        "'frustrated_positive', 'frustrated_negative', 'balanced_positive', 'balanced_negative'.")
        
    if highlight_edges and highlight_edges not in ['all', 'frustrated', 'balanced', 
                          'positive', 'negative',
                          'frustrated_positive', 'frustrated_negative',
                          'balanced_positive', 'balanced_negative']:
        raise Exception("InputError: show_edges must be one of 'all' (default), 'balanced', 'frustrated', 'positive', 'negative', "+
                        "'frustrated_positive', 'frustrated_negative', 'balanced_positive', 'balanced_negative'.")
        
        
    
    if not edges_color:
        edges_color = {'balanced_positive': 'cornflowerblue',#'blue',
                       'balanced_negative': 'red', #'red',
                       'frustrated_positive': 'blue', #'cyan',
                       'frustrated_negative': 'maroon'#'yellow'
                       }
    
    if type(edges_color) == dict:
        
        kinds = ['balanced_positive', 'balanced_negative', 'frustrated_positive', 'frustrated_negative']     
        if any([k not in edges_color for k in kinds]):
            raise Exception("Value Error: 'edges_color' must contain a color for each of {}.".format(', '.join(kinds)))
                
        import draw_edges_batches
        draw_edges_batches._draw_signed_networkx_edges(G, ax, pos, limits, edge_alpha, linewidth, edges_linestyle,
                                                       edges_color, show_edges, highlight_edges, outliers)
        
    else:
        try:
            _ = edges_color[0]
            assert len(edges_color) == len(G.edges())                        
        except:
            raise Exception("ValueError: if iterable, edges_color must be sized after the number of edges in G.")
            
        import draw_edges_iter
        draw_edges_iter._draw_signed_networkx_edges(G, ax, pos, limits, edge_alpha, linewidth, edges_linestyle,
                                                       edges_color, show_edges, highlight_edges, outliers)
        
                
                
                
def _draw_signed_networkx_nodes(G, ax, pos, 
                                node_size=40, 
                                node_color='black', 
                                node_shape='o', 
                                node_alpha=1.0, 
                                border_color = 'white', 
                                border_width = 1,
                                outliers = []):
    
    """
    Draws the nodes of G.

    Required arguments:
    ----------
    *G*:
        A networkx connected, undirected signed Graph, with edges equals to either -1 or +1.
        
    *ax*:
        The ax to draw the edge on.
        
    *pos*:
        A dict, with the x and y coordinates for each node in G. An entry is shaped like { node_id : Point( x , y ) }
    
    *node_size*:
        Either a numeric (default = 300) or an iterable. In case of iterable, it must be sized after the number of nodes in G.
        
    *node_color*:
        Either a string (default = black) or an iterable. In case of iterable, it must be sized after the number of nodes in G.
           
    *node_shape*:
        Either a char (default = 'o') or an iterable. In case of iterable, it must be sized after the number of nodes in G.
        
    *node_aplha*:
        A float. Alpha of the node to be drawn.
    
    *border_color*:
        Either a string (default = white) or an iterable. In case of iterable, it must be sized after the number of nodes in G.
        
    *border_width*:
        Either a numeric (default = 1) or an iterable. In case of iterable, it must be sized after the number of nodes in G.
     
    *outliers*:
        A list of nodes id, outliers not to be shown. Do not print arcs that involve these outliers.
        
    """
    
    import pandas as pd
    
    posx = [pos[node].x if node in pos else 0 for node in G.nodes()]
    posy = [pos[node].y if node in pos else 0 for node in G.nodes()]
    
    nodes = pd.DataFrame()
    nodes['node'] = [n for n in G.nodes()]
    nodes['posx'] = posx
    nodes['posy'] = posy
    nodes['color'] = node_color
    nodes['size'] = node_size
    nodes['border_color'] = border_color
    nodes['alpha'] = node_alpha
    nodes['border_width'] = border_width
    nodes['marker'] = node_shape
    
    nodes = nodes.loc[~nodes['node'].isin(outliers),]
    
    for mark in set(nodes['marker']):
        tmp = nodes.loc[nodes['marker'] == mark,]
        
        ax.scatter(tmp['posx'], tmp['posy'], facecolor = tmp['color'], edgecolor = tmp['border_color'], alpha = node_alpha,
               s = tmp['size'], linewidth = tmp['border_width'], marker = mark, zorder = 5)
        
    

 



    
def _setup_axes1(fig, angle, left, right, bottom, up, ax = None, rect = None):
    """
    Stacks a rotated axes over the main one.

    Required arguments:
    ----------
    *fig*:
        The figure that the axes are in.
        
    *angle*:
        A numeric limited in [-15, +15]. The angle of rotation of the canvas.
        
    *left*:
        A numeric. Leftmost point of the printable area.
    
    *right*:
        A numeric. Rightmost point of the printable area.
        
    *bottom*:
        A numeric. Lowest point of the printable area.
           
    *up*:
        A numeric. Highest point of the printable area.
     
    
    Return:
    -----------
    Two axes: a background one (ax1) and a rotated one (ax).
    The plot will be displayed on the rotated ax.
        
    """
    
    import mpl_toolkits.axisartist.floating_axes as floating_axes
    from mpl_toolkits.axisartist.grid_finder import MaxNLocator
    from matplotlib.transforms import Affine2D

    # Define height to width ratio
    vert = up - bottom
    hor = right - left
    ratio = vert / hor
    
    # Create rotated and scaled canvas
    tr = Affine2D().scale(4*ratio, 4).rotate_deg(angle)

    # Rotated canvas is in the center tile of a 3x3 grid, limited by extremes
    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(left, right, bottom, up),
        grid_locator1=MaxNLocator(nbins=4),
        grid_locator2=MaxNLocator(nbins=4))


    if not rect:
        rect = 111
    
    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    ax = ax1.get_aux_axes(tr)
    
    for axisLoc in ['top','left','right', 'bottom']:
        ax1.axis[axisLoc].set_visible(False)
        ax.axis[axisLoc].set_visible(False)
        
       
    
    return ax, ax1


def _draw_ax(fig, ax, limits, angle, show_rotation, rect, theme, least_eigenvalue):
    
    # Printable area
    minX, maxX, maxY = limits
    angle = angle if show_rotation else 0
    
    # Get plot extremes
    m = maxX if maxX > -minX else -minX
    
    if maxY == 0:
        maxY = 2

    left, right = (-m * 1.2, m * 1.2)
    up, bottom = ( maxY * 1.8, -maxY * 1.6)
    
    # Get rotated plot
    if not fig:
        fig = plt.figure(figsize = (10, 8))
    ax, ax1 = _setup_axes1(fig, angle, left, right, up, bottom, ax = ax, rect = rect)
    
    for figax in fig.axes:
        figax.axis('off')
        
    # Draw new axes
    if theme == 'dark':
        ccolor = '#AEAEAE'
        fig.set_facecolor('#252525')
    else:
        ccolor = 'black'
        
    plt.axvline(x = 0, color = ccolor, zorder = 2)
    ax.plot([left, right], [0, 0], color = ccolor)
        
    # Annotate mu
    if show_rotation:
        plt.annotate(s = 'v = {}'.format(least_eigenvalue), xy = ( plt.axis()[1] * .05, plt.axis()[3] * .75), fontsize = 15, color = ccolor)
        
        
    ax.tick_params(labeltop=False, labelbottom=False, labelleft=False)
    
    return fig, ax
    
    

def draw_signed_networkx(G,
                         ax = None,
                         rect = None,
                         fig = None,
                         node_size = 40,
                         node_alpha = .6,
                         edge_alpha = .6,
                         node_color='black', 
                         node_shape='o',
                         border_color = 'white', 
                         border_width = 1,
                         edges_color = None,
                         edges_linestyle = '-',
                         edge_linewidth = 1,
                         compact = False,
                         show_rotation = True,
                         show_edges = None,
                         highlight_edges = None,
                         remove_n_outliers = 0,
                         sort_by = None,
                         normalize = False,
                         jittering = 0,
                         margin = 0,
                         scale = 'linear',
                         theme = 'default'):
    
    """
    Draw a connected, undirected and signed network G.

    Required arguments:
    ----------
    *G*:
        The connected, undirected and signed networkx Graph to draw.
        
    *node_size*:
        Either a numeric or an iterable. If iterable, it must be sized after the number of nodes in G. 
        If not iterable, default is 40. The size of each (all) node(s).
        
    *node_alpha*:
        Either a numeric or an iterable. If iterable, it must be sized after the number of nodes in G. 
        If not iterable, default is 0.6. The alpha of each (all) node(s).
        
    *node_color*:
        Either a string or an iterable. If iterable, it must be sized after the number of nodes in G. 
        If not iterable, default is 'black'. The color of each (all) node(s).
    
    *node_shape*:
        Either a string or an iterable. If iterable, it must be sized after the number of nodes in G. 
        If not iterable, default is 'o'. The shape of each (all) node(s) (see: Matplotlib scatter markers).
    
    *border_color*:
        Either a string or an iterable. If iterable, it must be sized after the number of nodes in G. 
        If not iterable, default is 'white'. The color of the border of each (all) node(s).
    
    *border_width*:
        Either a numeric or an iterable. If iterable, it must be sized after the number of nodes in G. 
        If not iterable, default is 1. The width of the border of each (all) node(s).
    
    *edges_color*:
        Either a dict or an iterable. If iterable, it must be sized after the number of edges in G. 
        If a dict, it must contain a color for each kind of edge:
            balanced_positive, balanced_negative, frustrated_positive, frustrated_negative.
        Default is
            {
            'balanced_positive': 'cornflowerblue',
            'balanced_negative': 'red', 
            'frustrated_positive': 'blue', 
            'frustrated_negative': 'maroon'
            }
    
    *edge_linestyle*:
        Either a string or an iterable. If iterable, it must be sized after the number of edges in G. 
        If not iterable, default is '-' for continuous lines. The style of each (all) edge(s) (see: Matplotlib linestyles).
    
    *edge_linewidth*:
        Either a numeric or an iterable. If iterable, it must be sized after the number of edges in G. 
        If not iterable, default is 1. The width of each (all) edge(s).
     
    *show_rotation*:
        A boolean. If True, x-axis will be rotated towards the partition of nodes with more nodes into.
        A label will report the least eigenvalue, as a proxy for a frustration index of the Graph.
        
    *show_edges*:
        A string, one of "frustrated", "balanced", "frustrated_positive", "frustrated_negative", "balanced_positive", "balanced_negative",
        or "all". What kind of edges to show, while the other edges won't be drawed. 
        Default is "all". 'show_edges' and 'highlight_edges' are mutually exclusive.
        
    *highlight_edges*:
        A string, one of "frustrated", "balanced", "frustrated_positive", "frustrated_negative", "balanced_positive", "balanced_negative",
        or "all". What kind of edges to show regularly, while the other edges will be drawed in grey, with opacity = 0.01.
        Default is "all". 'show_edges' and 'highlight_edges' are mutually exclusive.
        
    *remove_n_outliers*:
        Remove n outliers nodes and arcs. Default is n = 0 (does not remove any node).
        
    *sort_by*:
        A string. An attribute of nodes in the Graph G.
        
     *normalize*:
        A boolean. If True, x-positions will be normalized to [-1, 1].
        
    *jittering*:
        A scalar. How much to jitter nodes' position. Default is 0, suggested is 0.002.
        
    *margin*:
        A scalar. It creates an empty belt around y = 0, in order to separate positive and negative points.
        
    *theme*:
        One of "default" or "dark".
        
    *scale*:
        One of "linear" or "log". Change the scale of x-axis.
        
    
    Returns:
    ------------
    
    *fig*:
        The matplotlib Figure object.
        
    *ax*:
        The matplotlib ax object.
        
    *pos*:
        A dict, each item a Point(x, y), representing the coordinates of the nodes.
            
    """
    
    
    # Get node positions
    pos, limits, angle, least_eigenvalue, outliers = nodes_coordinates(G, compact, sort_by, remove_n_outliers, normalize, margin, jittering, scale)
    
    # Printable area
    fig, ax = _draw_ax(fig, ax, limits, angle, show_rotation, rect, theme, least_eigenvalue)
    
    
    
    # Draw edges
    _draw_signed_networkx_edges(G, ax, pos, edge_alpha = edge_alpha, limits = limits, 
                                edges_color = edges_color,
                                edges_linestyle = edges_linestyle, linewidth = edge_linewidth,
                                show_edges = show_edges, highlight_edges = highlight_edges,
                                outliers = outliers)
    
    # Draw nodes
    _draw_signed_networkx_nodes(G, ax, pos, node_size = node_size, node_alpha = node_alpha, 
                                node_color = node_color, node_shape = node_shape, 
                                border_color = border_color, outliers = outliers)
        
    
    
    
    return fig, ax, pos
    

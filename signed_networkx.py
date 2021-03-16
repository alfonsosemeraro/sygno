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
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np
from matplotlib.collections import PatchCollection

#import line_profiler

__author__ = """Alfonso Semeraro (alfonso.semeraro@gmail.com)"""
__all__ = ['draw_signed_networkx',
           '_setup_axes1',
           '_draw_signed_networkx_nodes',
           '_draw_signed_networkx_edges',
           '_red_link',
           '_diag_blue_asc',
           '_diag_blue_desc',
           '_vert_link',
           '_horiz_blue',
           '_get_positions',
           '_get_L_matrix']


def _get_L_matrix(G) -> np.array:
    
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

    try:
        assert nx.is_connected(G) and not nx.is_directed(G)
    except:
        raise('Bad input: network G must be connected and undirected.')
        
        
    ## L = D - A
    
    # Computing A
    A = nx.adjacency_matrix(G, nodelist=G.nodes(), weight='weight')
    try:
        assert A.min() in [-1, 0] and A.max() in [0, 1]
    except:
        raise('Bad input: edge weights must be +1 or -1.')
    
    # Computing D
    D = zeros(A.shape)
    for i in range(D.shape[0]):
        D[i,i] = sum([abs(x) for x in A[i, :]]).sum()
           
    L = D - A
    
    return L




def _get_positions(G) -> tuple:
    
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
    *pos*:
        A dictionary with all the nodes positions. An entry is formatted as { node_id: Point( x, y ) }
        
    *minX*:
        A float. Minimal x-coordinate of any point in the plot. It will be used in order to define the printable area.
    
    *maxX*:
        A float. Maximal x-coordinate of any point in the plot. It will be used in order to define the printable area.
    
    *maxY*:
        A float. Maximal y-coordinate of any point in the plot. It will be used in order to define the printable area.
        
    *rot*:
        A float. How many degrees the x-axis will be rotated given the number of nodes in each partition.
        
    *least_eigenvalue*:
        A float. The least eigenvalue will be printed as a label, as additional information.
        
    """
    
    Point = namedtuple('Point', ['x', 'y'])

    # L = D - A
    L = _get_L_matrix(G)
    
    # least_eigenvector
    least_eigenvalue, least_eigenvector = eigsh(L, k=1, which='SM', return_eigenvectors=True)
    least_eigenvalue = round(least_eigenvalue[0], 4)
    
    # Computing position of each node as follows:
    pos = {}
    pos_y = {}
    
    # Limits of printable area
    minX = None
    maxX = None
    maxY = 0
    
    # Number of nodes in left and right side of the plot
    left = right = 0
    
    for i, node in enumerate(G.nodes()):
        
        # Get x coordinate
        x = round(least_eigenvector[i][0], 5)
        
        # Y position is dependent on X:
        # If there is no other point with the same X, then Y = 0
        # Else Y++
        if x not in pos_y.keys():
            pos_y[x] = 0
        else:
            pos_y[x] = pos_y[x] + 1
            
        # Update position dict
        p = Point(x, pos_y[x])
        pos[node] = p
        
        # Update limits:
        
        # min X
        if not minX or p.x < minX:
            minX = p.x
        
        # max X
        if not maxX or p.x > maxX:
            maxX = p.x
            
        # max Y
        if not maxY or p.y > maxY:
            maxY = p.y
            
        # count either left or right side
        if x < 0:
            left += 1
        elif x > 0:
            right += 1
          
        # rotation is a function of how many nodes are in each side
        # rotation is bound to [-15, +15] degrees
        try:
            rot = (left / (left + right)) * 30 - 15
        except:
            rot = 0
        
    return pos, minX, maxX, maxY, rot, least_eigenvalue





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
        p1.y != p2.y
        p1.x < p2.x
        
    """
    
    Point = namedtuple('Point', ['x', 'y'])
    
    # computing midpoints
    H = (p2.x - p1.x) / 2
    K = H + limits[2] / 8 # makes red edges wired and bundle, below 0
    
    mid1 = Point(p1.x, 0 - K)
    mid2 = Point(p1.x + H, 0 - K)
    mid3 = Point(p2.x, 0 - K)
    
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


def _draw_signed_networkx_edges(G, ax, pos, limits, edge_alpha = 1, 
                                linewidth = .6, edge_linestyle = '-',
                                positive_edges_color = 'steelblue', 
                                negative_edges_color = '#ff3255',
                                edges_color = None,
                                show_edges = 'all'):
    
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
    
    *positive_edges_color*:
        A string. Default is 'steelblue', but it can be customized. 
        
    *negative_edges_color*:
        A string. Default is '#ff3255', but it can be customized.
        
    *edge_linestyle*:
        A string. Default is '-', but it can be customized according to matplotlib linestyles.
        
    *show_edges*:
        A string, one of "frustrated", "balanced", "frustrated_positive", "frustrated_negative", "balanced_positive", "balanced_negative",
        or "all". Default is "all".
        
    """
    
    patches = []
    colors = [] if not edges_color else edges_color
    
    for source, target, w in list(G.edges(data = True)):
    
        weight = w['weight']
    
        p1, p2 = [pos[source], pos[target]]
        
        
        # Display only frustrated | balanced | both edges
        
        if (np.sign(p1.x) != np.sign(p2.x)) and weight == 1:
            kind = 'frustrated_positive'
        elif (np.sign(p1.x) == np.sign(p2.x)) and weight == -1:
            kind = 'frustrated_negative'
        elif weight == 1:
            kind = 'balanced_positive'
        else:
            kind = 'balanced_negative'
        
    
        if (show_edges != 'all') and (show_edges not in kind):
            continue
        
        
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
                
                if not edges_color:
                    colors.append(positive_edges_color)    
                patches.append(patch)
                    

            # E-i: same coordinates but not friends, red with horizontal internal bundling
            elif weight == -1:
                if p1.x < 0:
                    patch = _vert_link(ax, p1, p2, orient = 'right', limits = limits)
                else:
                    patch = _vert_link(ax, p1, p2, orient = 'leeft', limits = limits)
                    
                if not edges_color:
                    colors.append(negative_edges_color)       
                patches.append(patch)


        else:

            # E+e: different coordinates and friends, blue with vertical-upper bundling
            if weight == 1:
                
                if p1.y == p2.y:
                    patch = _horiz_blue(ax, p1, p2, limits = limits)
                
                elif p1.y < p2.y:
                    patch = _diag_blue_asc(ax, p1, p2)
            
                else:
                    patch = _diag_blue_desc(ax, p1, p2)
                    
                if not edges_color:
                    colors.append(positive_edges_color)       
                patches.append(patch)
                
                

            # E-e: different coordinates but not friends, red with vertical-lower bundling
            elif weight == -1:

                # p1 must be the left point, p2 must be the right point
                if p1.x > p2.x:
                    p1, p2 = p2, p1
                
                patch = _red_link(ax, p1, p2, limits = limits)
                
                if not edges_color:
                    colors.append(negative_edges_color)       
                patches.append(patch)
         
    
    patches = PatchCollection(patches, facecolor = 'none', linewidth = linewidth, edgecolor = colors,  match_original=True, linestyle = edge_linestyle, alpha = edge_alpha)
    ax.add_collection(patches)
                
                
                
                
def _draw_signed_networkx_nodes(G, ax, pos, node_size=40, node_color='black', node_shape='o', node_alpha=1.0, border_color = 'white', border_width = 1):
    
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
     
        
    """
    

    posx = [pos[node].x for node in G.nodes()]
    posy = [pos[node].y for node in G.nodes()]
    
    ax.scatter(posx, posy, facecolor = node_color, edgecolor = border_color, alpha = node_alpha,
               s = node_size, linewidth = border_width, marker = node_shape, zorder = 2)



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
                         positive_edges_color = None,
                         negative_edges_color = None,
                         edges_color = None,
                         edge_linestyle = '-',
                         edge_linewidth = 1,
                         show_rotation = True,
                         show_edges = 'all'):
    
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
    
    *positive_edge_color*:
        Either a string or an iterable. If iterable, it must be sized after the number of edges in G. 
        If not iterable, default is 'steelblue'. The color of each (all) positive edge(s).
    
    *negative_edge_color*:
        Either a string or an iterable. If iterable, it must be sized after the number of edges in G. 
        If not iterable, default is '#ff3255'. The color of each (all) negative edge(s).
        
    *edge_colors*:
        An iterable. Contains one color for each edge. If positive_edge_color or negative_edge_color is initialised,
        this parameter is ignored.
    
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
        or "all". Default is "all".
            
    """
    
    # Get node positions
    pos, minX, maxX, maxY, angle, least_eigenvalue = _get_positions(G)
    angle = angle if show_rotation else 0
    
    # Get plot extremes
    m = maxX if maxX > -minX else -minX
    
    if maxY == 0:
        maxY = 2
        
    left, right = (-m * 1.2, m * 1.2)
    up, bottom = (- maxY / 1.8, maxY * 1.3)
    
    # Get rotated plot
    if not fig:
        fig = plt.figure(figsize = (10, 8))
    ax, ax1 = _setup_axes1(fig, angle, left, right, up, bottom, ax = ax, rect = rect)
    
    for figax in fig.axes:
        figax.axis('off')
    
    
    if positive_edges_color or negative_edges_color:
        edges_color = None

    if not positive_edges_color:
        positive_edges_color = 'steelblue'
        
    if not negative_edges_color:
        negative_edges_color = '#ff3255'
        
        
    if show_edges not in ['all', 'frustrated', 'balanced', 
                          'frustrated_positive', 'frustrated_negative',
                          'balanced_positive', 'balanced_negative']:
        raise Exception("Value error: show_edges must be one of 'all' (default), 'balanced', 'frustrated', 'frustrated_positive',"+
                        " 'frustrated_negative', 'balanced_positive', 'balanced_negative'.")
    
    # Draw edges and nodes
    limits = (minX, maxX, maxY)
    _draw_signed_networkx_edges(G, ax, pos, edge_alpha = edge_alpha, limits = limits, 
                                positive_edges_color = positive_edges_color, 
                                negative_edges_color = negative_edges_color,
                                edges_color = edges_color,
                                edge_linestyle = edge_linestyle, linewidth = edge_linewidth,
                                show_edges = show_edges)
    
    _draw_signed_networkx_nodes(G, ax, pos, node_size = node_size, node_alpha = node_alpha, 
                                node_color = node_color, node_shape = node_shape, 
                                border_color = border_color)
        
    
    # Draw new axes
    plt.axvline(x = 0, color = 'black')
    ax.plot([left, right], [0, 0], color = 'black')
        
    # Annotate mu
    if show_rotation:
        plt.annotate(s = 'v = {}'.format(least_eigenvalue), xy = ( plt.axis()[1] * .05, plt.axis()[3] * .75), fontsize = 15)
        
        
    ax.tick_params(labeltop=False, labelbottom=False, labelleft=False)
    
    return fig, ax, pos
    

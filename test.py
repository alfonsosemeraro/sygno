#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 13:32:56 2020

@author: alfonso
"""

import pandas as pd
import networkx as nx
from signed_networkx import draw_signed_networkx
import matplotlib.pyplot as plt
import itertools
from glob import glob

#df = pd.read_csv('congress.csv', header = None)
#df.columns = ['source', 'target', 'weight']
#
#df = df.loc[df['source'] != df['target']]
#df = df.drop_duplicates(subset = ['source', 'target'])
#
#G = nx.from_pandas_edgelist(df, 'source', 'target', ['weight'])
#    
#draw_signed_networkx(G)
#plt.savefig('congress.png', dpi = 300, bbox_inches = 'tight');



def read_congress(file, art):
    df = pd.read_csv(file, header=1)
    yay = list(df.loc[df['vote'] == 'Yea', 'person'].values)
    nay = list(df.loc[df['vote'] == 'Nay', 'person'].values)
    
    opposites = itertools.product(yay, nay)
    opposites = pd.DataFrame(opposites, columns = ['source', 'target'])
    opposites['weight'] = -1
    opposites['count'] = -1
    
    yay = itertools.combinations(yay, 2)
    yay = pd.DataFrame(yay, columns = ['source', 'target'])
    yay['weight'] = 1
    yay['count'] = 1
    
    nay = itertools.combinations(nay, 2)
    nay = pd.DataFrame(nay, columns = ['source', 'target'])
    nay['weight'] = 1
    nay['count'] = 1
    
    return art.append(opposites).append(yay).append(nay)


art = pd.DataFrame()

## GLOB
for file in glob('arthur/*.csv'):
    art = read_congress(file, art)
    art = art.groupby(['source', 'target', 'weight']).sum().reset_index()


## DRAW
G = nx.from_pandas_edgelist(art, 'source', 'target', ['weight'])

#largest_cc = max(nx.connected_components(G), key=len)
#S = G.subgraph(largest_cc).copy() 
    
draw_signed_networkx(G, show_edges = 'balanced_negative', theme = 'dark')
plt.savefig('congress.png', dpi = 300, bbox_inches = 'tight');


"""

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
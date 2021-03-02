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



df = pd.read_csv('congress.csv', header = None)
df.columns = ['source', 'target', 'weight']

df = df.loc[df['source'] != df['target']]
df = df.drop_duplicates(subset = ['source', 'target'])

G = nx.from_pandas_edgelist(df, 'source', 'target', ['weight'])
    
draw_signed_networkx(G)
plt.show();


"""

def draw_signed_networkx(G,
                         node_size = 40,
                         node_alpha = .6,
                         edge_alpha = .6,
                         node_color='black', 
                         node_shape='o',
                         border_color = 'white', 
                         border_width = 1,
                         positive_edges_color = 'steelblue',
                         negative_edges_color = '#ff3255',
                         edge_linestyle = '-',
                         edge_linewidth = 1,
                         show_rotation = True):
    
    
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
    
    *edge_linestyle*:
        Either a string or an iterable. If iterable, it must be sized after the number of edges in G. 
        If not iterable, default is '-' for continuous lines. The style of each (all) edge(s) (see: Matplotlib linestyles).
    
    *edge_linewidth*:
        Either a numeric or an iterable. If iterable, it must be sized after the number of edges in G. 
        If not iterable, default is 1. The width of each (all) edge(s).
     
    *show_rotation*:
        A boolean. If True, x-axis will be rotated towards the partition of nodes with more nodes into.
        A label will report the least eigenvalue, as a proxy for a frustration index of the Graph.
            
"""
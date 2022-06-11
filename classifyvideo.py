# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 15:38:10 2019

@author: User
"""
from openpyxl import load_workbook
import os

wb=load_workbook('video.xlsx')

ws=wb.active


path=[]
pathh=[]
for row in ws.rows:
    
    if row[0].value ==None:
       continue
    if 'Chunk' in row[0].value:
        row0=row[0].value
    if ('Chunk' not in row[0].value) :
        if 'n' not in str(row[1].value):
            
            row1=str(row[1].value).split(',')
            
            row2=str(row[2].value).split(',')
            if 'n' not in row2:
               for _ in row2:
                   if _ in row1:
                      row1.remove(_)
            
            path1=['/media/jinnliu/My Passport/'+row0+'/'+row[0].value +'/'+k+'/camera320.h5' for k in row1]
            path2=['/media/jinnliu/My Passport/'+row0+'/'+row[0].value +'/'+k+'/video.hevc' for k in row1]
            path.extend(path1)
            pathh.extend(path2)

path_v=[]
#for p in path:
  #  if p.startswith('Chunk_10') or p.startswith('Chunk_9'):
   #     path_v.append(p)
#for v in path_v:
 #   path.remove(v)
#path_test=['/mnt/usb/'+v for v in path_v]
#path_train=['/mnt/usb/'+vv for vv in path]


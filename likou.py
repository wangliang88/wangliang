# -*- coding: utf-8 -*-
"""
Created on Sun May 16 16:46:37 2021

@author: wl255
"""
# class solution(object):
#     def isprime(n):
#         if n ==3 or n==2 :
#             return True
#         for i in range(2,int(pow(n,0.5))+1,1):
#             if n%i==0 :
#                 return False
#         return True
#     def nprime(n):
#         num=0
#         prime=[]
#         if n < 2:
#             return num 
#         for i in range(2,n+1):
#             if solution.isprime(i):
#                 num+=1
#                 prime.append(i)
#         print(prime)
#         return num
#         #print(num)
# out=solution.nprime(6)
# # print(out)
# def gcd(n,m):
#     n=int(n)
#     m=int(m)
#     if m>n:
#         temp=n
#         n=m
#         m=temp
#     if n%m==0:
#         return m
#     else:
#          return gcd(m,n%m)
    
# def n_gcd(n):
#     if len(n)==1:
#         return -1
#     out=n[0]
#     for i in range(1,len(n)):
#         out=gcd(out,n[i])
#     return out

# n=list((input("请输入需要求最大公约数的数组:\n").strip().split()))
# #print(len(n))
# print(n_gcd(n))

# def rmdup(n):
#     lengh=len(n)
#     low=0
#     fast=1
#     if lengh==0 :
#         return lengh
#     n[0]=int(n[0])
#     for fast in range(lengh):
#         if int(n[low])!=int(n[fast]):
#             low=low+1
#             n[low]=int(n[fast])
#         fast=fast+1
#     for i in range(lengh-1,low,-1):
#         n.pop(n[i])
#     print(n)
#     return low+1
# n=list(((input("请输入需要处理的数组:\n").strip()).split()))
# print(rmdup(n))
# import os
# import cv2
# data_dir = "c"
# dir_path=os.listdir(data_dir)
# for i in range(len(dir_path)):
#     image_path=os.path.join(data_dir, dir_path[i])
#     hr=cv2.imread(image_path)
#     height,width=hr.shape[:2]
#     lr=cv2.resize(hr,height/4,width/4,interpolation=cv2.INTER_CUBIC)
#     hr_path=
# def lengthOfLongestSubstring(s: str) -> int:
#         if not s:return 0
#         left = 0
#         lookup = set()
#         n = len(s)
#         max_len = 0
#         cur_len = 0
#         for i in range(n):
#             cur_len += 1
#             while s[i] in lookup:
#                 lookup.remove(s[left])
#                 left += 1
#                 cur_len -= 1
#             if cur_len > max_len:max_len = cur_len
#             lookup.add(s[i])
#         print(lookup)
#         return max_len
# def longesthuiwen(s:str):
   
#     if not s:
#         return 0
#     windows=set()
#     left=0
#     right=0
#     max_len=0
#     cur_len=0
#     while(right<len(s)):
        
#         cur_len=cur_len+1
#         while (s[right] in windows):
#             windows.remove(s[left])
#             left=left+1
#             cur_len=cur_len-1
#         if cur_len > max_len :
#             max_len=cur_len
#         windows.add(s[right])
#         right=right+1
#     return max_len
# in_=input('请输入一组字符串：\n')
# out=longesthuiwen(in_) 
# print(out)           
# def kuohao(s:str) ->bool :
#     lengh=len(s)
#     stack=[]
#     if lengh%2==1:
#         return False
#     _dict={')':'(',']':'[','}':'{'}
#     for i in range(lengh):
#         if s[i]=='(' or s[i]=='[' or s[i]=='{':
#             stack.append(s[i])
#         if s[i]==')' or s[i]==']' or s[i]=='}' and len(stack) !=0:
#             if stack.pop()!=_dict[s[i]]:
#                 return False
#     return len(stack)==0
# in_=input('请输入一组括号：\n')
# print(in_)
# out=kuohao(in_) 
# print(out)
# def lefterfenchazhao(s,target):
#     right=len(s)-1
#     left=0
#     while(left<=right):
#         mid=left+int((right-left)/2)
#         if s[mid]==target:
#             right=mid-1 
#         if s[mid]>target:
#             right=mid-1
#         if s[mid]<target:
#             left=mid+1
#     if left>len(s)-1 or s[left]!=target:
#         return -1
#     return left

# a=[1,3,4,6,7,7,8,8,8,9,11,15,18]
# out=lefterfenchazhao(a, 18)
# print(out)
# def twotarget(s,target):
#     left=0
#     ans=[]
#     right=len(s)-1
#     while(left<right):
#         if s[left]+s[right]<target:
#             left=left+1
#         if s[left]+s[right]>target:
#             right=right-1
#         if s[left]+s[right]==target:
#             ans.append([left,right])
#             left=left+1
#             right=right-1
            
#             while(left<right and left-1>=0 and s[left]==s[left-1]):
#                 left=left+1
#             while(left<right and right+1<len(s) and s[right]==s[right+1]):
#                 right=right-1
#     return ans
# a=[0,1,3,4,6,7,8,8,10,13,15,15,17,18,21,22]
# out=twotarget(a, 21)
# print(out)
# def dajiajieshe(s.):
#     lengh=len(s)
#     dp=list()
#     i=2
#     if lengh==0:
#         return -1
#     if lengh==1:
#         return s
#     if lengh==2:
#         return max(s[0],s[1])
#     dp.append(s[0])
#     dp.append(max(s[0],s[1]))
#     while(i<lengh):
#         dp.append(max((s[i]+dp[i-2]),dp[i-1]))
#         i=i+1
#     return dp[-1]
# a=[1,2,4,5,5]
# out=dajiajieshe(a)
# print(out)
# import torch
# a=torch.tensor([3,5,6,7,8,9])
# b=torch.max(a,)
# print(b[0].shape)
# import time
# arry=[4,2,7,10,8,11,14,20,1,5,9,21,15]
# def bubblesort(arry):
#     start_t=time.time()
#     lengh=len(arry)
#     for i in range(0,lengh,1):
#         for j in range (0,lengh-i-1):
#             if  arry[j]>arry[j+1]:
#                 arry[j],arry[j+1]=arry[j+1],arry[j]
#                 time.sleep(0.01)
#     time_len=time.time()-start_t
#     return arry,time_len


#buout_arry,burun_time=bubblesort(arry)
#print("冒泡排序好的数组为：",out_arry)
#print("冒泡排序时间为：",burun_time)

# def selectsort(arry):
#     sestart_t=time.time()
#     lengh=len(arry)
#     for i in range(0,lengh-1,1):
#         for j in range(i+1,lengh ,1):
#             if arry[i]>arry[j]:
#                 arry[i],arry[j]=arry[j],arry[i]
#                 time.sleep(0.1)
#     time_len=time.time()-sestart_t
#     return arry,time_len
 
# #seout_arry,serun_time=selectsort(arry)
# #print("选择排序好的数组为：",out_arry)
# #print("选择排序时间为：",serun_time)   

# def insertsort(arry):
# #    start_t=time.time()
#     lengh=len(arry)
#     for i in range(1,lengh,1):
#         temp=arry[i]
#         j=i-1
#         while(j>=0 and arry[j]>temp):
#             arry[j+1]=arry[j]
#             j=j-1
#         if j<i-1:
#             arry[j+1]=temp
#     return arry
        
# #out_arry=insertsort(arry)
# #print("插入排序好的数组为：",out_arry)
# #print("选择排序时间为：",serun_time)  

# def quicksort(arry):
    
#     if len(arry)>=2:
#         mid=arry[len(arry)//2]
#         left=[]
#         right=[]
#         arry.remove(mid)
#         for num in arry:
#             if num>mid:
#                 right.append(num)
#             else:
#                 left.append(num)
#         return quicksort(left)+[mid]+quicksort(right)
#     else:
#         return arry
# out_arry=quicksort(arry)
# print("插入排序好的数组为：",out_arry)


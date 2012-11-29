== Readme ==
This readme briefly explains how to use the codes. Our DTW code requires four parameters in this syntax: 	
    UCR_DTW   Data_File  Query_File  M   R		 
    or
    UCR_ED    Data_File  Query_File  M   
		 
Input Explanation,
    UCR_DTW (or UCR_ED):  the executable file of our algorithm
    Data_File:  the data file containing a long time series
    Query_File: the query file containing the query, Q
    M:  length of the query time series. It usually is |Q|, but can be smaller.
    R:  size of warping windows. The value is in range 0-1, e.g., R=0.05 means windows of size +/-5%. 	

Output Explanation,
    Location: starting location of the nearest neighbor of the given query, of size M, in the data file. Note that location starts from 0.
	Distance: the distance between the nearest neighbor and the query.
	Data Scaned: number of data in the input data file.
	Pruned by LB: the number of subsequence can be prunned by using that LB (in percentage).
	DTW Calculation: the number DTW calculation has been made in total.
	Note that, when approximate one million data points have been read, one dot "." will be displayed on screen (only in UCR_DTW).
	
== Example 1 ==
	----------------------------------------------------
	C:\> UCR_DTW Data.txt Query.txt 128 0.05
	..
	Location : 756562
	Distance : 3.77562
	Data Scanned : 1000000
	Total Execution Time : 1.584 sec

	Pruned by LB_Kim    :  67.97%
	Pruned by LB_Keogh  :  22.25%
	Pruned by LB_Keogh2 :   9.32%
	DTW Calculation     :   0.46%
	----------------------------------------------------


== Example 2 ==
	----------------------------------------------------
	C:\> UCR_ED Data.txt Query.txt 128
	Location : 347236
	Distance : 7.03705
	Data Scanned : 1000000
	Total Execution Time : 1.029 sec
	----------------------------------------------------


== Example 3 ==
	----------------------------------------------------
	C:\> UCR_DTW Data.txt Query2.txt 128 0.10
	..
	Location : 430264
	Distance : 3.7907
	Data Scanned : 1000000
	Total Execution Time : 3.182 sec

	Pruned by LB_Kim    :  22.68%
	Pruned by LB_Keogh  :  35.98%
	Pruned by LB_Keogh2 :  36.60%
	DTW Calculation     :   4.74%
	----------------------------------------------------



== Fairness of Usage ==
Please reference this paper in your paper as 
Thanawin Rakthanmanon, Bilson Campana, Abdullah Mueen, Gustavo Batista, Brandon Westover, Qiang Zhu, Jesin Zakaria, Eamonn Keogh (2012). Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping SIGKDD 2012.



== Disclaimer ==
This UCR Suite software is copyright protected © 2012 by Thanawin Rakthanmanon, Bilson Campana, Abdullah Mueen, Gustavo Batista, and Eamonn Keogh.
Unless stated otherwise, all software is provided free of charge. As well, all software is provided on an "as is" basis without warranty of any kind, express or implied. Under no circumstances and under no legal theory, whether in tort, contract, or otherwise, shall Thanawin Rakthanmanon, Bilson Campana, Abdullah Mueen, Gustavo Batista, or Eamonn Keogh be liable to you or to any other person for any indirect, special, incidental, or consequential damages of any character including, without limitation, damages for loss of goodwill, work stoppage, computer failure or malfunction, or for any and all other damages or losses.
If you do not agree with these terms, then you you are advised to not use the software.



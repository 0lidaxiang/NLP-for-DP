#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import numpy

dataArr = numpy.loadtxt('./data/data.txt')
answerTemp = numpy.loadtxt('./data/answer.txt')
answerArr = answerTemp.reshape((answerTemp.shape[0],1))

print(dataArr + answerArr)

# data_test_1 , answer_test_1
# data_test_2 , answer_test_2

# dataArr = numpy.loadtxt('./data/data_test_1.txt')
# answerTemp = numpy.loadtxt('./data/answer_test_1.txt')
# answerArr = answerTemp.reshape((answerTemp.shape[0],1))
# print(dataArr + answerArr)
#
# dataArr = numpy.loadtxt('./data/data_test_2.txt')
# answerTemp = numpy.loadtxt('./data/answer_test_2.txt')
# answerArr = answerTemp.reshape((answerTemp.shape[0],1))
# print(dataArr + answerArr)

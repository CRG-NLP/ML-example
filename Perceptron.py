__author__ = 'zhf'
#coding=utf-8
# Perceptron
# Date: 2016-01-01

import copy
from matplotlib import pyplot as plt
from matplotlib import animation

# init
training_data = [[(3,3),1],[(4,3),1],[(1,1),-1]]

w = [0,0]   #weight
b = [0]     #bias
step = 1    #learning rate
history = []

"""
update parameters using stochastic gradient descent
parameter item: an item which is classified into wrong class
return: NULL
"""
def update(item):
    global w,b,history
    w[0] += step*item[1]*item[0][0]
    w[1] += step*item[1]*item[0][1]
    b[0] += step*item[1]
    print w,b
    history.append([copy.copy(w),copy.copy(b)])

"""
calculate the item classified result.
y_i*(w*x_i+b)<=0 wrong
parameter item: training_data
return: classified result
"""
def calculate(item):
    res = 0;
    for i in range(len(item[0])):
        res += w[i]*item[0][i]
    res += b[0]
    res *= item[1]
    return res

"""
check if the classifier can correctly classify all data
parameter item: NULL
return: correct or not
"""
def check():
    flag = False
    for item in training_data:
        if calculate(item) <= 0:
            update(item)
            flag = True
    if not flag:
        print "Result: w:" + str(w) +"b" + str(b)
    return flag

# main function
if __name__ == "__main__":
    for i in range(1000):
        if not check():
            break

# set up the figure
    fig = plt.figure()
    ax = plt.axes(xlim=(0,6),ylim=(0,6))
    line, = ax.plot([],[],lw=2)
    label = ax.text([],[],'')
# init function for base frame
    def init():
        line.set_data([],[])
        x1, y1, x2, y2 = [],[],[],[]
        for item in training_data:
            if item[1]>0:
                x1.append(item[0][0])
                y1.append(item[0][1])
            else:
                x2.append(item[0][0])
                y2.append(item[0][1])
        plt.plot(x1,y1,'bo',x2,y2,'rx')
        plt.axis([-6,6,-6,6])
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Perceptron')
        return line, label
    def animate(i):
        global ax, line, label
        w = history [i][0]
        b = history [i][1]
        # hyperplane: w[0]*x_i+w[1]*y_i+b; y_i=-(w[0]*x_i+b)/w[1]
        if w[1] == 0:
            return line, label
        x1 = -6
        y1 = -(w[0]*x1 + b[0]/w[1])
        x2 = 6
        y2 = -(w[0]*x2 + b[0]/w[1])
        line.set_data([x1,x2],[y1,y2])
        x0 = 0
        y0 = -(w[0]*x0 + b[0])/w[1]
        label.set_text(history[i])
        label.set_position([x0,y0])
        return line,label
    # animation function
    def animate(i):
        global ax, line, label
        w = history [i][0]
        b = history [i][1]

        if w[1] == 0:
            return line, label
        x1 = -6
        y1 = -(w[0]*x1 + b[0])/w[1]
        x2 = 6
        y2 = -(w[0]*x2 + b[0])/w[1]
        line.set_data([x1,x2],[y1,y2])
        x0 = 0
        y0 = -(w[0]*x0 + b[0])/w[1]
        label.set_text(history[i])
        label.set_position([x0,y0])
        return line,label
    #call the animator

    print history
    anim = animation.FuncAnimation(fig,animate,init_func=init,frames=len(history),interval=1000)
    plt.show()
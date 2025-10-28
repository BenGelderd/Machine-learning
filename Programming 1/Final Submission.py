import numpy as np
from matplotlib import pyplot as plt
from numpy import cos, sin
from math import sqrt, sin, cos, tan

#1b
def draw_line(m, c, x0, e11):
    x_lst = np.linspace(x0, x0 + e11, 20)
    y_lst = [i*m + c for i in x_lst]
    plt.plot(x_lst, y_lst, '-')
    
#2

def draw_unit_circle(thecolour):
    lst = np.linspace(0, 2*np.pi, 100)
    x = []
    y = []
    for i in lst:
        x.append(cos(i))
        y.append(sin(i))
    plt.plot(x, y, color=thecolour)
#     plt.gca().set_aspect('equal', 'box')
    
#3b
def draw_circle(circ, thecolour):
    x = circ[0][0]
    r = circ[1]
    y = circ[0][1]
    lst = np.linspace(0, 2*np.pi, 100)
    ylst = []
    xlst = []
    for i in lst:
        xlst.append((r*cos(i)+x))
        ylst.append(r*sin(i)+y)
    plt.plot(xlst, ylst, color=thecolour)
    plt.gca().set_aspect('equal', 'box')

#4
def show_six_circles_example():
    j = [0, 1, 2, 3, 4, 5]
    colours = ['blue', 'red', 'green', 'yellow', 'black', 'purple']
    for i in j:
        draw_circle([ [cos(i*np.pi/3), sin(i*np.pi/3)], 0.5 ], colours[i])
#     plt.show()

#5b
def draw_ellipse(n,m,a,b):
    lst = np.linspace(0, 2*np.pi, 100)
    ylst = []
    xlst = []
    for i in lst:
        xlst.append((a*cos(i))+n)
        ylst.append((b*sin(i))+m)
    plt.plot(xlst, ylst, 'red')
    
#6b
def draw_tangent(circ, xy, d, thecolour):
    x = circ[0][0]
    r = circ[1]
    y = circ[0][1]
    m = 0
    #tangent equtation exceptions
    if xy[1] - y == 0:
        m = 'y_line'
    if xy[0] - x == 0:
        m = 'x_line'
    else:
        m = -1*(xy[0]-x)/(xy[1]-y)
        
    lst = np.linspace(0, 2*np.pi, 1000000) #increasing the intervals will lead tomore specific points by which rounding points is not neccessary but will increase the time of the function
    ylst = []
    xlst = []
    t_ylst = [] 
    for i in lst:
        xlst.append((r*cos(i)+x))
        ylst.append(r*sin(i)+y)
   #check if (x,y) lies on circle:
    val = 0
    for i in range(len(lst)):
        #rounding values from list to 3 d.p as xy are likely to be floats with 3 or less d.p
        if [round(xlst[i],3), round(ylst[i], 3)] == xy:
            val = 1

    if val == 1:
        #generate a list of x and y values such that the length of the tangent == d
        if m == 'y_line':
            x_tlst = np.linspace(xy[0], xy[0], 1000)
            y_tlst = np.linspace(xy[1]-d, xy[1]+d, 1000)
        if m == 'x_line':
            x_tlst = np.linspace(xy[0]-d, xy[0]+d, 1000)
            y_tlst = np.linspace(xy[1], xy[1], 1000)
        else:
            theta = np.arctan(m)
            x_tlst = np.linspace(xy[0]-(0.5*d*cos(theta)), xy[0]+(0.5*d*cos(theta)), 1000)
            y_tlst = [m*(i - xy[0]) + xy[1] for i in x_tlst]
        draw_circle(circ, thecolour)
        plt.plot(x_tlst, y_tlst, color=thecolour)
        
    else:
        print(False)
        return False
    
#7
def show_tangent_example():
    circ = [[-1, 3], 2]
    thecolour = 'red'
    draw_circle(circ, thecolour)
    draw_tangent(circ, [0.2, 4.6], 4, thecolour)
#     plt.show()

#8b
def xcircles_and_radicalaxis(r, R, c, thecolours):
    try:
        xval = 0.5*c + (r**2-R**2)/(2*c)
        y = sqrt(r**2 - xval**2)
        x_lst = np.linspace(xval, xval, 1000)
        y_lst = np.linspace(-y, y, 1000)
        draw_circle([[0,0], r], thecolours[0])
        draw_circle([[c,0], R], thecolours[1])
        plt.gca().set_aspect('equal', 'box')
        plt.plot(x_lst, y_lst, thecolours[2])
    except ZeroDivisionError:
        draw_circle([[0,0], r], thecolours[0])
        draw_circle([[c,0], R], thecolours[1])
    except ValueError:
        draw_circle([[0,0], r], thecolours[0])
        draw_circle([[c,0], R], thecolours[1])
        
#9b
def new_xcircles_and_radicalaxis(r, R, c, thecolours):
    try:
        xval = 0.5*c + (r**2-R**2)/(2*c)
        y = sqrt(r**2 - xval**2)
        x_lst = np.linspace(xval, xval, 1000)
        y_lst = np.linspace(-y, y, 1000)
#have removed the drawing circle phase
        return [[x_lst[i], y_lst[i]] for i in range(1000)]
    except ZeroDivisionError:
        return False
    except ValueError:
        return False

def gencircles_and_radicalaxis(circ1, circ2, thecolours):
    x1 = circ1[0][0]
    y1 = circ1[0][1]
    x2 = circ2[0][0]
    y2 = circ2[0][1]
    draw_circle(circ1, thecolours[0])
    draw_circle(circ2, thecolours[1])
    #circles after translation
    circ1_0 = [[0,0], circ1[1]]
    circ2_0 = [[x2-x1, y2-y1], circ2[1]]
    #rotation
    theta = 0
    if circ2_0[0][0] == 0:
        theta = np.pi/2
    else:
        theta = np.arctan(circ2_0[0][1]/circ2_0[0][0])
    c = circ2_0[0][0]*cos(theta) + circ2_0[0][1]*sin(theta)
    try:
        lsts = new_xcircles_and_radicalaxis(circ1[1], circ2[1], c, thecolours)
        x_lst = [lsts[i][0] for i in range(len(lsts))]
        y_lst = [lsts[i][1] for i in range(len(lsts))]
    #converting back to original plane position
        X_lst = [x_lst[i]*cos(theta) - y_lst[i]*sin(theta)+x1 for i in range(len(x_lst))]
        Y_lst = [y_lst[i]*cos(theta) + x_lst[i]*sin(theta)+y1 for i in range(len(x_lst))]
        plt.plot(X_lst, Y_lst, thecolours[2])
    except TypeError:
        #if no solution
        return False
    
#10b
def determine_tangent(circ, xy):
    x = circ[0][0]
    r = circ[1]
    y = circ[0][1]
    m = ''
    #tangent equtation exceptions
    if xy[1] - y == 0:
        m = 'y_line'
    if xy[0] - x == 0:
        m = 'x_line'
    else:
        m = -1*(xy[0]-x)/(xy[1]-y)
    return m

def single_bumper(theta, circ, e11, circ_col, line_col):
    draw_circle(circ, circ_col)
    #draw the initial track
    x0 = circ[0][0]
    y0 = circ[0][1]
    r = circ[1]
    try:
        a1 = (2*x0 + 2*y0*tan(theta) - sqrt((2*x0+2*y0*tan(theta))**2 - (4*(1/cos(theta)**2)*(y0**2+x0**2-r**2))))/(2*1/(cos(theta)**2))
        a2 = (2*x0 + 2*y0*tan(theta) + sqrt((2*x0+2*y0*tan(theta))**2 - (4*(1/cos(theta)**2)*(y0**2+x0**2-r**2))))/(2*1/(cos(theta)**2))
        b1 = a1*tan(theta)
        b2 = a2*tan(theta)
        a = ''
        b = ''
        if sqrt(a1**2 + b1**2) < sqrt(a2**2 + b2**2) :
            a = a1
            b = b1
        else:
            a = a2
            b = b2
        
        initial_xlst = np.linspace(0, a, 1000)
        initial_ylst = [ i*tan(theta) for i in initial_xlst]
        plt.plot(initial_xlst, initial_ylst, color=line_col)
        #find gradient of tangent to determine rebound angle
        m = determine_tangent(circ, [a,b])
        gamma = 0
        if m == 'y_line':
            gamma = 0
        elif m == 'x_line':
            gamma = 0
        else:
            gamma = np.arctan(m)
       
        if x0> 0 :
        #     conditions when collision occurs on upper region of the circle or negative y region
            if b > y0 and (y0 + r) > 0:
                rebound_angle = gamma - theta
                beta = np.pi/2 - rebound_angle
                G = e11-(sqrt(a**2 + b**2))
                x_final = a - G*cos(beta)
                y_final = b + G*sin(beta)
                xlst = np.linspace(a, x_final, 1000)
                ylst = np.linspace(b, y_final, 1000)
                
                
            elif round(b, 2) == y0:
                #   determine whether pinball is deflected back or forward
                p = np.arctan(y0/x0)
                if theta > p: #deflected back
                    rebound_angle = abs(gamma) + theta
                if theta <= p: #deflected forward
                    rebound_angle = theta - gamma
                #plot rebound track
                beta = rebound_angle + np.pi/2
                G = e11-(sqrt(a**2 + b**2))
                x_final = a + G*sin(beta)
                xlst = np.linspace(a, x_final, 1000)
                ylst = [ (b-(cos(beta)*(i-a)/sin(beta))) for i in xlst ]
               

            elif (y0 + r) < 0 and a < x0:
                G = e11-(sqrt(a**2 + b**2))
                rebound_angle = theta - gamma - np.pi
                beta = np.pi - rebound_angle 
                if b < y0:
                    x_final = a + G*sin(beta)
                else:
                    x_final = a - G*sin(beta)
                xlst = np.linspace(a, x_final, 1000)
                ylst = [ (b-(cos(beta)*(i-a)/sin(beta))) for i in xlst ]
                
                
                
            elif (y0 + r) < 0 and a > x0:
                rebound_angle = gamma - theta
                beta = np.pi/2 + rebound_angle 
                G = e11-(sqrt(a**2 + b**2))
                x_final = a + G*sin(beta)
                xlst = np.linspace(a, x_final, 1000)
                ylst = [ (b-(cos(beta)*(i-a)/sin(beta))) for i in xlst ]
               
                
            else:
            #   determine whether pinball is deflected back or forward
                p = np.arctan(y0/x0)
                if theta > p: #deflected back
                    rebound_angle = abs(gamma) + theta
                if theta <= p: #deflected forward
                    rebound_angle = theta - gamma
                #plot rebound track
                beta = np.pi/2 - rebound_angle
                G = e11-(sqrt(a**2 + b**2))
                x_final = a + G*sin(beta)
                xlst = np.linspace(a, x_final, 1000)
                ylst = [ (b-(cos(beta)*(i-a)/sin(beta))) for i in xlst ]
                
        
        elif x0 < 0: # if in negative x region
            #     conditions when collision occurs on upper region of the circle or negative y region
            if b > y0 and a >= x0:
                G = e11-(sqrt(a**2 + b**2))
                if y0 > 0:
                    rebound_angle = np.pi - theta + gamma
                    beta = np.pi/2 - rebound_angle
                    x_final = a + G*sin(beta)
                else:
                    rebound_angle = theta - np.pi + abs(gamma)
                    beta = np.pi/2 - rebound_angle
                    x_final = a - G*sin(beta)
                    print('here')
                xlst = np.linspace(a, x_final, 1000)
                ylst = [ (b-(cos(beta)*(i-a)/sin(beta))) for i in xlst ]
                
                
    
                
            elif b > y0 and a < x0:
                rebound_angle = theta - (gamma + np.pi)
                beta  = np.pi/2 - rebound_angle
                G = e11-(sqrt(a**2 + b**2))
                x_final = a - G*sin(beta)
                xlst = np.linspace(a, x_final, 1000)
                ylst = [ (b-(cos(beta)*(i-a)/sin(beta))) for i in xlst ]
                print('here')
               
            
            elif b < y0 and a > x0:
                rebound_angle = np.pi - theta + gamma
                beta  = np.pi/2 - rebound_angle
                G = e11-(sqrt(a**2 + b**2))
                x_final = a + G*sin(beta)
                xlst = np.linspace(a, x_final, 1000)
                ylst = [ (b-(cos(beta)*(i-a)/sin(beta))) for i in xlst ]
                
                
            else:
            #   determine whether pinball is deflected back or forward
                p = np.arctan(y0/(-1*x0))
                if (theta-np.pi/2) <= p: #deflected back
                    rebound_angle = abs(gamma) + theta
                if (theta-np.pi/2) > p: #deflected forward
                    rebound_angle = np.pi - (theta - gamma)
                #plot rebound track
                beta = np.pi/2 - rebound_angle
                G = e11-(sqrt(a**2 + b**2))
                x_final = a + G*sin(beta)
                xlst = np.linspace(a, x_final, 1000)
                ylst = [ (b-(cos(beta)*(i-a)/sin(beta))) for i in xlst ]
                
    
        plt.plot(xlst, ylst, color=line_col)
        
    except ValueError: #no collision
        x_end = e11*cos(theta)
        y_end = e11*sin(theta)
        xl = np.linspace(0, x_end, 1000)
        yl = np.linspace(0, y_end, 1000)
        plt.plot(xl, yl, color=line_col)
        

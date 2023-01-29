#!/usr/bin/python3
#coding=utf-8
from socket import IP_MULTICAST_LOOP
from time import sleep
import sys
from turtle import right
import cv2
import rospy
import time
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from cv2 import SIFT

class moveRobot(object): # Se crea una clase para realizar los movimientos del robot
    def __init__(self):
        self.ctrl_c = False 
        self.velPublisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1) # Se crea el publicador

    def publishOnceCmdVel(self, cmd): # Se crea la funcion de publicacion
        while not self.ctrl_c:
            connections = self.velPublisher.get_num_connections()
            if connections > 0:
                self.velPublisher.publish(cmd)
                #rospy.loginfo("Cmd Published")
                break
            else:
                pass
    def stop(self): # Se crea una funcion de detenido
        #rospy.loginfo("shutdown time! Stop the robot")
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.publishOnceCmdVel(cmd)

    def move(self, movingTime, linear1, angular1): # Se crea una funcion de movimiento
        cmd = Twist()
        cmd.linear.x = linear1
        cmd.angular.z = angular1
        
        self.publishOnceCmdVel(cmd)
        time.sleep(movingTime)

class laneDetect():
    def __init__(self):
        """************************************************************
        ** Initialise variables
        ************************************************************"""
        self.moveObject = moveRobot()
        self.cvImage = []
        self.bridge = CvBridge()
        
        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        
    
        # Initialise subscribers
        #self.imageSub = rospy.Subscriber("/camera/image_compensated", Image, self.imageCallback)
        
        # Creamos el topico en el que publicaremos la imagen resultado
        self.imagePub = rospy.Publisher("opencvTopic", Image, queue_size=1)
        #self.bridge = CvBridge()  # Creamos un objeto para realizar la conversion de la imagen
        # Creamos el subcriptor al topico de la camara
        self.imageSub = rospy.Subscriber("/camera/image", Image, self.callback)
        #self.cvImage = 0

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""

                
                   
    def callback(self, msg):
        try:
            self.cvImage = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.detectLane()
        except CvBridgeError as e:
            self.get_logger().info("Turtlebot3 image is not captured.")    

    def comparate_img(self, original, image_to_compare):
        # 1) comparan las dos imagenes
        if original.shape == image_to_compare.shape:
            difference = cv2.subtract(original, image_to_compare)
            b, g, r = cv2.split(difference)
    
        # 2) similitud
        shift = cv2.xfeatures2d.SIFT_create()
        kp_1, desc_1 = shift.detectAndCompute(original, None)
        kp_2, desc_2 = shift.detectAndCompute(image_to_compare, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_1, desc_2, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.9*n.distance:
                good_points.append(m)

        number_keypoints = 0
        if (len(kp_1) <= len(kp_2)):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)


        return result
           
            
    def detectLane(self):
        im_1 = cv2.imread("izquierda.png")  #base de datos imagenes
        im_2 = cv2.imread("derecha.png")  #base de datos imagenes
        im_3 = cv2.imread("parar.png")  #base de datos imagenes
        im_4 = cv2.imread("construccion.png")  #base de datos imagenes
        im_5 = cv2.imread("interseccion.png")  #base de datos imagenes
        im_6 = cv2.imread("tunel.png")  #base de datos imagenes
        im_7 = cv2.imread("parqueo.png")  #base de datos imagenes
        heigth, width = self.cvImage.shape[:2]
        cutImg = self.cvImage[heigth*2//3:heigth,:]
        #print(heigth,width)
        whiteFraction, whiteLane, whiteLaneRGB = self.maskWhiteLane(cutImg)
        yellowFraction, yellowLane, yellowLaneRGB = self.maskYellowLane(cutImg)
        
        
        try:
            if yellowFraction > 50:
                self.leftFitx, self.leftFit = self.slidingWindow(yellowLane, 'left')
                self.movAvgLeft = np.array([self.leftFit])
                

            if whiteFraction > 50:
                self.rightFitx, self.rightFit = self.slidingWindow(whiteLane, 'right')
                self.movAvgRight = np.array([self.rightFit])
                
            
        except:
            pass
        
        
        final, central = self.makeLane(cutImg)      
        center_gray = cv2.cvtColor(central, cv2.COLOR_BGR2GRAY)
        hg, wd = center_gray.shape
        centerright = center_gray [:,wd*2//3:wd]
        centerLeft = center_gray[:,:wd//3]
       
        nonzeroright = centerright.nonzero()
        nonzeroright = np.array(nonzeroright[0])
        
        nonzeroLeft = centerLeft.nonzero()
        nonzeroLeft = np.array(nonzeroLeft[0])
        
        v_Left = np.mean(nonzeroLeft)
        v_right = np.mean(nonzeroright)
        v_right = np.nan_to_num(v_right)/25
        v_Left = np.nan_to_num(v_Left)/25
        print(v_Left)
        print(v_right)
      
       
        self.v_real = v_Left - v_right
            
    
        gray1= cv2.cvtColor(self.cvImage,cv2.COLOR_BGR2GRAY)
        sift1 = cv2.SIFT_create()
        kp = sift1.detect(gray1,None)
        img_final=cv2.drawKeypoints(gray1,kp,self.cvImage)
        


        im_left = self.comparate_img(im_1, img_final)
        im_right = self.comparate_img(im_2, img_final)
        im_stop = self.comparate_img(im_3, img_final)
        im_construccion = self.comparate_img(im_4, img_final)
        im_interseccion = self.comparate_img(im_5, img_final)
        im_tunel = self.comparate_img(im_6, img_final)
        im_parqueo = self.comparate_img(im_7,img_final)
        
        print(im_left)
        print(im_right)
        print(im_stop)
        print(im_construccion)
        print(im_interseccion)
        print(im_tunel)
        print(im_parqueo)

       
      
    
        
        #print(self.v_real)        
        cv2.imshow('image', cutImg)
        cv2.imshow('white lane', whiteLaneRGB)
        cv2.imshow('yellow lane', yellowLaneRGB)
        cv2.imshow('final lane', final)
        cv2.imshow('central', central)
        cv2.imshow('izquierda', centerLeft)
        cv2.imshow('derecha', centerright)
        cv2.waitKey(1)
        
        #self.moveObject.move(0.01, 0.1, self.v_real)
        time.sleep(0.01)
        
        
    def maskWhiteLane(self, image):
        
        
 #segmentacion de la imagen 
        lowerWhite = np.array([200, 200, 200])
        upperWhite = np.array([255, 255, 255])
        mask = cv2.inRange(image, lowerWhite, upperWhite)
        res = cv2.bitwise_and(image, image, mask=mask)

        fractionNum = np.count_nonzero(mask)

        return fractionNum, mask, res

    def maskYellowLane(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


        lowerYellow = np.array([20, 255, 255])
        upperYellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lowerYellow, upperYellow)
        res = cv2.bitwise_and(image, image, mask = mask)

        fractionNum = np.count_nonzero(mask)

        return fractionNum, mask, res

    def slidingWindow(self, imgW, leftOrRight):
        histogram = np.sum(imgW[imgW.shape[0] // 2:, :], axis=0)
        # Creamos una imagen de salida donde observar el resultado
        outImg = np.dstack((imgW, imgW, imgW)) * 255
        # Buscamos el pico o punta de las mitades derechas e izquierdas en el histograma
        # Estas van a ser el punto de partida para las lineas izquierda y derecha
        midpoint = np.int(histogram.shape[0] // 2)

        if leftOrRight == 'left':
            laneBase = np.argmax(histogram[:midpoint])
        elif leftOrRight == 'right':
            laneBase = np.argmax(histogram[midpoint:]) + midpoint
        
        # Definimos la cantidad de ventanas deslizantes
        nwindows = 20
        
        # Definimos el alto de las ventanas
        windowHeight = np.int(imgW.shape[0] / nwindows)
        
        # Identificamos las posiciones xy de todos los pixeles en la imagen que no tienen un valor de 0
        nonzero = imgW.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Posiciones actuales que seran actualizadas para cada ventana
        xCurrent = laneBase

        # Definimos el ancho de las ventanas +/- margin
        margin = 50

        # Definimos la cantidad minima de pixeles encontrados para re ubicar la ventana
        minpix = 50

        # Creamos una lista vacia para recibir los indices de los pixeles que componen el carril
        laneInds = []

        # Pasamos por las ventanas una por una
        for window in range(nwindows):
            # Identificamos los limites en las coordenadas xy
            winYLow = imgW.shape[0] - (window + 1) * windowHeight
            winYHigh = imgW.shape[0] - window * windowHeight
            winXLow = xCurrent - margin
            winXHigh = xCurrent + margin

            # Dibujamos las ventanas en la imagen de salida o visualziacion
            cv2.rectangle(outImg, (winXLow, winYLow), (winXHigh, winYHigh), (0, 255, 0), 2)
            
            # Identificamos los pixeles que no son 0 en las coordenadas xy dentro de la ventana
            goodlaneInds = ((nonzeroy >= winYLow) & (nonzeroy < winYHigh) & (nonzerox >= winXLow) & (
                nonzerox < winXHigh)).nonzero()[0]

            # Agregamos estos indices a la lista
            laneInds.append(goodlaneInds)

            # Si se encuentra una mayor cantidad de minpix entonces se reubica el centro de la ventana
            # en su posicion media
            if len(goodlaneInds) > minpix:
                xCurrent = np.int(np.mean(nonzerox[goodlaneInds]))
                
            
        cv2.imshow('imgW', imgW)
        cv2.imshow('outImg', outImg)
        cv2.waitKey(1)
            

        # Concatenamos el arreglo de los indices
        laneInds = np.concatenate(laneInds)

        # Extraemos la posicion de los pixeles de la linea
        x = nonzerox[laneInds]
        y = nonzeroy[laneInds]

        # Ajustamos una funcion polinomica de segundo orden a cada una
        try:
            laneFit = np.polyfit(y, x, 2)
            laneFitBef = laneFit
        except:
            laneFit = laneFitBef

        # Generamos los valores de las coordenadas xy para graficar
        ploty = np.linspace(0, imgW.shape[0] - 1, imgW.shape[0])
        laneFitx = laneFit[0] * ploty ** 2 + laneFit[1] * ploty + laneFit[2]
        
        return laneFitx, laneFit

    
    def makeLane(self, cvImage):
        # Creamos una imagen donde dibujar las lineas
        warpZero = np.zeros((cvImage.shape[0], cvImage.shape[1], 1), dtype=np.uint8)

        colorWarp = np.dstack((warpZero, warpZero, warpZero))
        colorWarpLines = np.dstack((warpZero, warpZero, warpZero))
        colorWarpLinescenter = np.dstack((warpZero, warpZero, warpZero))

        # Creamos un vector de las posibles coordenadas en y que puede tomar el carril
        ploty = np.linspace(0, cvImage.shape[0] - 1, cvImage.shape[0])

        # Obtenemos los puntos de la linea izquierda del carril y lo dibujamos sobre la imagen
        ptsLeft = np.array([np.flipud(np.transpose(np.vstack([self.leftFitx, ploty])))])
        cv2.polylines(colorWarpLines, np.int_([ptsLeft]), isClosed=False, color=(0, 0, 255), thickness=25)
   
        # Obtenemos los puntos de la linea derecha del carril y lo dibujamos sobre la imagen
        ptsRight = np.array([np.transpose(np.vstack([self.rightFitx, ploty]))])
        cv2.polylines(colorWarpLines, np.int_([ptsRight]), isClosed=False, color=(255, 255, 0), thickness=25)
  
        # Calculamos el centro del carril con base a los ajustes realizados sobre las lineas izquierda y derecha
        centerx = np.mean([self.leftFitx, self.rightFitx], axis=0)
        pts = np.hstack((ptsLeft, ptsRight))
        ptsCenter = np.array([np.transpose(np.vstack([centerx, ploty]))])
        
        # Dibujamos el carril sobre los limites encontrados con anterioridad
        cv2.polylines(colorWarpLines, np.int_([ptsCenter]), isClosed=False, color=(0, 255, 255), thickness=12)
        cv2.polylines(colorWarpLinescenter, np.int_([ptsCenter]), isClosed=False, color=(0, 255, 255), thickness=12)
        cv2.fillPoly(colorWarp, np.int_([pts]), (0, 255, 0))
 
        # Combinamos los resultados con la imagen original
        final = cv2.addWeighted(cvImage, 1, colorWarp, 0.2, 0)
        final = cv2.addWeighted(final, 1, colorWarpLines, 1, 0)
    
        return final, colorWarpLinescenter
        
def main(args):
    ldexp = laneDetect()  # Iniciamos la clase
    rospy.init_node('laneDetect', anonymous=True)  # Creamos el nodo
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

import numpy as np
import cv2
from operator import itemgetter

#abre a captura de vídeo. parâmetro 0 = webcam
cap = cv2.VideoCapture('v2.MP4')

c1=[0,0,0]
c2=[0,0,0]
c3=[0,0,0]
c4=[0,0,0]
c5=[0,0,0]
c6=[0,0,0]
c7=[0,0,0]
c8=[0,0,0]
c9=[0,0,0]
c10=[0,0,0]
c11=[0,0,0]

def ligaPontos(p1,p2):
    cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (0,0,0))
    
def desenhaBoneco(pontos):
    
    pontoTroncoMeio = 0
    k = 0
    #Liga Tronco
    backup = 0
    pontos = sorted(pontos, key=itemgetter(1))
    for i in range(len(pontos)):
        if (pontos[i][2] == 0):
            cv2.circle(img, (pontos[i][0], pontos[i][1]), 5, (0,0,255), -1)
            k = k + 1
            if(k == 1):
                cv2.circle(img, (pontos[i][0], pontos[i][1]-30), 30, (0,0,0))
            if(k == 2):
                pontoTroncoMeio = pontos[i]
            if(backup != 0):
                ligaPontos(backup, pontos[i])
            backup = pontos[i]
            
    #liga braços e pernas
    k = 0
    bracoE = [img.shape[1]-1, 0 , 0]
    pernaE = [img.shape[1]-1, 0 , 0]
    bracoD = [0,0,0]
    pernaD = [0,0,0]
    for i in range(len(pontos)):
        if (pontos[i][2] == 1):
            cv2.circle(img, (pontos[i][0], pontos[i][1]), 5, (0,255,0), -1)
            k = k + 1
            if(k <= 2):
                ligaPontos(pontoTroncoMeio, pontos[i])
                if(pontos[i][0] < bracoE[0]):
                    bracoE = pontos[i]
                if(pontos[i][0] > bracoD[0]):
                    bracoD = pontos[i]
            else:
                ligaPontos(backup, pontos[i])
                if(pontos[i][0] < pernaE[0]):
                    pernaE = pontos[i]
                if(pontos[i][0] > pernaD[0]):
                    pernaD = pontos[i]
     
    #liga mão e pé esquerdo
    k = 0
    for i in range(len(pontos)):
        if (pontos[i][2] == 2):
            cv2.circle(img, (pontos[i][0], pontos[i][1]), 5, (255,0,0), -1)
            k = k + 1
            if(k == 1):
                ligaPontos(bracoE, pontos[i])
            else:
                ligaPontos(pernaE, pontos[i])
                
    
    #liga mão e pé direito
    k = 0
    for i in range(len(pontos)):
        if (pontos[i][2] == 3):
            cv2.circle(img, (pontos[i][0], pontos[i][1]), 5, (0,255,255), -1)
            k = k + 1
            if(k == 1):
                ligaPontos(bracoD, pontos[i])
            else:
                ligaPontos(pernaD, pontos[i])
#======================================================================================================
def findMaxValueCount(c):
    #retorna a área em pixels de um determinado contorno
    maxArea = cv2.contourArea(c[0])
    contourId = 0
    i = 0
    for cnt in c:
        if maxArea < cv2.contourArea(cnt):
            maxArea = cv2.contourArea(cnt)
            contourId = i
        i += 1
    #em cnt obtemos o contorno de maior área do conjunto de possíveis contornos
    cnt = c[contourId]
    return cnt, contourId,maxArea

def desenhaRetangulo(cntT, maxArea):
    #retorna um retângulo que envolve o contorno em questão
    x2,y2,w2,h2 = cv2.boundingRect(cntT)
    cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255,0), 2)
    center2 = (x2,y2)
    if(maxArea < 100.0):
        cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(0,0,255),3)
        return center2
    return (0,0)

while (True):
    #lê cada quadro e carrega na variável frame
    _, frame = cap.read()
    frame = cv2.flip(frame,1)

    #transforma a imagem de RGB para HSV
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #vermelho vermelho
    lowerRed = np.array([161, 200, 200])
    upperRed = np.array([179, 255, 255])

    #vermelho verde
    lowerGreen = np.array([45, 200, 200])
    upperGreen = np.array([75, 255, 255])

    #vermelho azul
    lowerBlue = np.array([94, 200, 200])
    upperBlue = np.array([126, 255, 255])

    #vermelho amarelo
    lowerYellow = np.array([20, 200, 200])
    upperYellow = np.array([30, 255, 255])

    #CFG TO LED RED
    mask = cv2.inRange(hsvImage, lowerRed, upperRed)
    result = cv2.bitwise_and(frame, frame, mask = mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)    
    _,gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    #CFG TO LED GREEN
    maskGreen = cv2.inRange(hsvImage, lowerGreen, upperGreen)
    resultGreen = cv2.bitwise_and(frame, frame, mask = maskGreen)
    grayGreen = cv2.cvtColor(resultGreen, cv2.COLOR_BGR2GRAY)    
    _,grayGreen = cv2.threshold(grayGreen, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contoursGreen, hierarchyGreen = cv2.findContours(grayGreen, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #CFG TO LED Blue
    maskBlue = cv2.inRange(hsvImage, lowerBlue, upperBlue)
    resultBlue = cv2.bitwise_and(frame, frame, mask = maskBlue)
    grayBlue = cv2.cvtColor(resultBlue, cv2.COLOR_BGR2GRAY)    
    _,grayBlue = cv2.threshold(grayBlue, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contoursBlue, hierarchyBlue = cv2.findContours(grayBlue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #CFG TO LED Yellow
    maskYellow = cv2.inRange(hsvImage, lowerYellow, upperYellow)
    resultYellow = cv2.bitwise_and(frame, frame, mask = maskYellow)
    grayYellow = cv2.cvtColor(resultYellow, cv2.COLOR_BGR2GRAY)    
    _,grayYellow = cv2.threshold(grayYellow, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contoursYellow, hierarchyYellow = cv2.findContours(grayYellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #se existir contornos do LED RED
    if contours:
        #verificadores
        v2=0
        v3=0
        
        #pontos
        c1=[0,0,0]
        c2=[0,0,0]
        c3=[0,0,0]
        
        cnt, idRemove,maxArea1 = findMaxValueCount(contours)
        del contours[idRemove]
        if(contours):
            v2=1
            cnt2, idRemove,maxArea2 = findMaxValueCount(contours)
            del contours[idRemove]
        if(contours):
            v3=1
            cnt3, idRemove,maxArea3 = findMaxValueCount(contours)
            del contours[idRemove]
                    
        #retorna um retângulo que envolve o contorno em questão
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255,0), 2)
        c1 = (x,y)        
        #printa a posicao do vermehlo
        #print ("centro_1 : ",c1)        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        
        #Desenha o retangulo do ponto vermelho
        if(v2==1 ):
            c2 = desenhaRetangulo(cnt2,maxArea2)
            #print("centro_2 : ", c2)
            
        #Desenha o retangulo do ponto vermelho
        if(v3==1 ):
            c3 = desenhaRetangulo(cnt3,maxArea3)
            #print("centro_3 : ", c3)

    #se existir contornos do LED green
    if contoursGreen:
        #verificadores
        v5=0
        v6=0
        v7=0
        
        #pontos
        c4=[0,0,0]
        c5=[0,0,0]
        c6=[0,0,0]
        c7=[0,0,0]
        
        cnt4, idRemove, maxArea4 = findMaxValueCount(contoursGreen)
        del contoursGreen[idRemove]
        if(contoursGreen):
            v5=1
            cnt5, idRemove,maxArea5 = findMaxValueCount(contoursGreen)
            del contoursGreen[idRemove]
        if(contoursGreen):
            v6=1
            cnt6, idRemove, maxArea6 = findMaxValueCount(contoursGreen)
            del contoursGreen[idRemove]
        if(contoursGreen):
            v7=1
            cnt7, idRemove, maxArea7 = findMaxValueCount(contoursGreen)
            del contoursGreen[idRemove]
                    
        #retorna um retângulo que envolve o contorno em questão
        x4,y4,w4,h4 = cv2.boundingRect(cnt4)
        cv2.rectangle(frame, (x4, y4), (x4 + w4, y4 + h4), (0, 255,0), 2)
        c4 = (x4,y4)        
        #printa a posicao do vermehlo
        #print ("centro_4 : ",c4)        
        cv2.rectangle(frame,(x4,y4),(x4+w4,y4+h4),(0,0,255),3)
        
        #Desenha o retangulo do ponto vermelho
        if(v5==1 ):
            c5 = desenhaRetangulo(cnt5,maxArea5)
            #print("centro_5 : ", c5)
            
        #Desenha o retangulo do ponto vermelho
        if(v6==1 ):
            c6 = desenhaRetangulo(cnt6,maxArea6)
           # print("centro_6 : ", c6)

        #Desenha o retangulo do ponto vermelho
        if(v7==1 ):
            c7 = desenhaRetangulo(cnt7,maxArea7)
            #print("centro_7 : ", c7)

##    #se existir contornos do LED Blue
    if contoursBlue:
        #verificadores
        v9=0
        
        #pontos
        c8=[0,0,0]
        c9=[0,0,0]
        
        cnt8, idRemove,maxArea8 = findMaxValueCount(contoursBlue)
        del contoursBlue[idRemove]
        if(contoursBlue):
            v9=1
            cnt9, idRemove,maxArea9 = findMaxValueCount(contoursBlue)
            del contoursBlue[idRemove]
                    
        #retorna um retângulo que envolve o contorno em questão
        x8,y8,w8,h8 = cv2.boundingRect(cnt8)
        cv2.rectangle(frame, (x8, y8), (x8 + w8, y8 + h8), (0, 255,0), 2)
        c8 = (x8,y8)        
        #printa a posicao do vermehlo
       # print ("centro_8 : ",c8)        
        cv2.rectangle(frame,(x8,y8),(x8+w8,y8+h8),(0,0,255),3)
        
        #Desenha o retangulo do ponto vermelho
        if(v9==1 ):
            c9 = desenhaRetangulo(cnt9,maxArea9)
            #print("centro_9 : ", c9)

    #se existir contornos do LED Yellow
    if contoursYellow:
        #verificadores
        v11=0
        
        #pontos
        c10=[0,0,0]
        c11=[0,0,0]
        
        cnt10, idRemove,maxArea10 = findMaxValueCount(contoursYellow)
        del contoursYellow[idRemove]
        if(contoursYellow):
            v11=1
            cnt11, idRemove,maxArea11 = findMaxValueCount(contoursYellow)
            del contoursYellow[idRemove]
                    
        #retorna um retângulo que envolve o contorno em questão
        x10,y10,w10,h10 = cv2.boundingRect(cnt10)
        cv2.rectangle(frame, (x10, y10), (x10 + w10, y10 + h10), (0, 255,0), 2)
        c10 = (x10,y10)        
        #printa a posicao do vermehlo
        #print ("centro_10 : ",c10)        
        cv2.rectangle(frame,(x10,y10),(x10+w10,y10+h10),(0,0,255),3)
        
        #Desenha o retangulo do ponto vermelho
        if(v11==1 ):
            c11 = desenhaRetangulo(cnt11,maxArea11)
            print("centro_11 : ", c11)

    img = np.zeros((600,600,3),  dtype="uint8")
    img = cv2.bitwise_not(img)

    

    lista = []
    lista = [ [c1[0], c1[1], 0],  [c2[0], c2[1], 0],  [c3[0], c3[1], 0],  [c4[0], c4[1], 1],   [c5[0], c5[1], 1], [c6[0], c6[1], 1],   [c7[0], c7[1], 1],    [c8[0], c8[1], 2],    [c9[0], c9[1], 2],  [c10[0], c10[1], 3],  [c11[0], c11[1], 3]]
    lista = [ [0, 0, 0],  [0, 0, 0],  [0, 0, 0],  [c4[0], c4[1], 1],   [c5[0], c5[1], 1], [c6[0], c6[1], 1],   [c7[0], c7[1], 1], [0,0, 2],    [0,0, 2],  [0,0, 3],  [0,0, 3]]

    desenhaBoneco( lista )
    cv2.imshow("a", img)
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

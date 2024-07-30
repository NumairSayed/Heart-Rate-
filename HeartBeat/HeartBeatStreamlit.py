import streamlit as st
import cv2
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PlotModule import LivePlot
import cvzone
import time
import numpy as np

# Streamlit App
st.title("Heart Rate Detection")

def detect():
    realWidth = 640   
    realHeight = 480  
    videoWidth = 160
    videoHeight = 120
    videoChannels = 3
    videoFrameRate = 15

    detector = FaceDetector()
    cap = cv2.VideoCapture(0)
    cap.set(3, realWidth)
    cap.set(4, realHeight)

    levels = 3
    alpha = 170
    minFrequency = 1.0
    maxFrequency = 2.0
    bufferSize = 150
    bufferIndex = 0

    plotY = LivePlot(realWidth, realHeight, [60, 120], invert=True)

    def buildGauss(frame, levels):
        pyramid = [frame]
        for level in range(levels):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def reconstructFrame(pyramid, index, levels):
        filteredFrame = pyramid[index]
        for level in range(levels):
            filteredFrame = cv2.pyrUp(filteredFrame)
        filteredFrame = filteredFrame[:videoHeight, :videoWidth]
        return filteredFrame

    font = cv2.FONT_HERSHEY_SIMPLEX
    loadingTextLocation = (30, 40)
    bpmTextLocation = (videoWidth//2, 40)

    firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
    firstGauss = buildGauss(firstFrame, levels+1)[levels]
    videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
    fourierTransformAvg = np.zeros((bufferSize))

    frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
    mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

    bpmCalculationFrequency = 10
    bpmBufferIndex = 0
    bpmBufferSize = 10
    bpmBuffer = np.zeros((bpmBufferSize))

    i = 0
    ptime = 0
    ftime = 0

    start_time = time.time()

    all_bpm_values = []

    # Create placeholders for the video and plot
    video_placeholder = st.empty()
    plot_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time = time.time() - start_time
        if elapsed_time >= 10:
            break

        frame, bboxs = detector.findFaces(frame, draw=False)
        frameDraw = frame.copy()
        ftime = time.time()
        fps = 1 / (ftime - ptime)
        ptime = ftime

        cv2.putText(frameDraw, f'FPS: {int(fps)}', (30, 440), 0, 1, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)
        if bboxs:
            x1, y1, w1, h1 = bboxs[0]['bbox']
            cv2.rectangle(frameDraw, bboxs[0]['bbox'], (255, 0, 255), 2)
            detectionFrame = frame[y1:y1 + h1, x1:x1 + w1]
            detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight))

            videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
            fourierTransform = np.fft.fft(videoGauss, axis=0)

            fourierTransform[mask == False] = 0

            if bufferIndex % bpmCalculationFrequency == 0:
                i = i + 1
                for buf in range(bufferSize):
                    fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
                hz = frequencies[np.argmax(fourierTransformAvg)]
                bpm = 78.0 * hz
                bpmBuffer[bpmBufferIndex] = bpm
                bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

            filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
            filtered = filtered * alpha

            filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
            outputFrame = detectionFrame + filteredFrame
            outputFrame = cv2.convertScaleAbs(outputFrame)

            bufferIndex = (bufferIndex + 1) % bufferSize
            outputFrame_show = cv2.resize(outputFrame, (videoWidth//2, videoHeight//2))
            frameDraw[0:videoHeight // 2, (realWidth-videoWidth//2):realWidth] = outputFrame_show

            bpm_value = bpmBuffer.mean()
            imgPlot = plotY.update(float(bpm_value))

            all_bpm_values.append(bpm_value)

            if i > bpmBufferSize:
                cvzone.putTextRect(frameDraw, f'BPM: {bpm_value}', bpmTextLocation, scale=2)
            else:
                cvzone.putTextRect(frameDraw, "Calculating BPM...", loadingTextLocation, scale=2)

            imgStack = cvzone.stackImages([frameDraw, imgPlot], 2, 1)

        else:
            imgStack = cvzone.stackImages([frameDraw, frameDraw], 2, 1)

        video_placeholder.image(imgStack, channels="BGR")

    average_bpm = sum(all_bpm_values) / len(all_bpm_values) if all_bpm_values else 0
    st.write(f"Average BPM: {average_bpm}")

    BPMstate = average_bpm >= 60 and average_bpm <= 100
    return BPMstate, average_bpm

if st.button('Start Heart Rate Detection'):
    BPMstate, average_bpm = detect()
    st.write(f"BPM State: {'Normal' if BPMstate else 'Abnormal'}, Average BPM: {average_bpm}")

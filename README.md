# TesisMicro
Reconocimiento y clasificación de patrones en imágenes de microscopia, parametrizadas según la Técnica de Micronúcleos.

Se utilizó transferencia de aprendizaje de la arquitectura VGG16 prevista en Keras. Mediante  modificaciones en las últimas capas, se  redujo el sobre-entrenamiento y adaptó el modelo al algoritmo creado para la detección de objetos en una imagen dada. 
El aprendizaje es un proceso de modificación de los pesos en respuesta a los estímulos presentes en la capa de entrada de una red neuronal, el pipeline que se creó no alcanza extraer puntualmente las características particulares de las imágenes de MN y por lo tanto, no se logró la adaptación correcta de los pesos. 

El fallo fundamental está en la construcción de un pipeline específico por clase de muestra. Sin embargo para probar su funcionalidad, se entrenó en paralelo al modelo VGG16MicroTesis con el conjunto de datos COCO, modificando las imágenes de entrada en la carpeta DATA. Con las modificaciones. se logró la clasificación eficiente de objetos con las etiquetas “árbol”, “persona”, “caballo” por ejemplo.  
Los siguientes pasos para avanzar con el proyecto son, sin lugar a dudas, dar mayor difusión a lo realizado a fin de que, con la exposición pública y a través de publicaciones científicas, se logre crear un dataset con los pipeline necesarios para detectar los MN y AN en las imágenes de microscopía.

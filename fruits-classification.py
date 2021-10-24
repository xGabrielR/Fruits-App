import numpy as np
import streamlit as st
import tensorflow as tf

from pandas import DataFrame
from tensorflow.keras import Model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.applications import MobileNetV2

class Fruits( object ):
    def __init__( self ) :
        self.model_weights = 'model/mobilenet.h5'

    def pre_html( self ):
        html='''
        <style>
        ::selection {
            color: #b950ff;
        }
        h1 {
            color: #7033ff;
            text-align: center;
        }
        .line {
            position: absolute;
            width: 135px;
            left: 200px;
            opacity: 15%;
            height: 20px;
            top: 45px;
            background: rgb(253, 250, 255 );
        }
        .line2 {
            position: absolute;
            width: 135px;
            left: 190px;
            opacity: 15%;
            height: 20px;
            top: 115px;
            background: rgb(253, 250, 255 );
        }
        @media screen and (max-width: 1100px) {
            .line2{
                top: 125px;
                left: 197px;
                width: 120px;
            }
        }
        @media screen and (max-width: 800px) {
            .line2{
                top: 120px;
                left: 197px;
                width: 120px;
            }
        }
        @media screen and (max-width: 700px) {
            .line{
                top: 35px;
                left: 205px;
                width: 120px;
            }
            .line2{
                top: 110px;
                left: 197px;
                width: 120px;
            }
        }
        @media screen and (max-width: 600px) {
            .line{
                top: 35px;
                left: 150px;
                width: 120px;
            }
            .line2{
                top: 110px;
                left: 142px;
                width: 120px;
            }
        }
        @media screen and (max-width: 500px) {
            .line{
                top: 35px;
                left: 120px;
                width: 120px;
            }
            .line2{
                top: 108px;
                left: 115px;
                width: 120px;
            }
        }
        </style>
        <section>
        <h1>🍎 Fruits & Vegetables</h1>
        <h1>Classification 🌽</h1>
        <div class='line'></div>
        <div class='line2'></div>
        </section>
        '''
        st.markdown( html,unsafe_allow_html=True )
        return None

    def model_preparation( self ):
        model_pre = MobileNetV2( input_shape=(224, 224, 3),
                                include_top = False,
                                weights = 'imagenet',
                                pooling = 'avg' )

        model_pre.trainable = False

        inputs = model_pre.input
        x = tf.keras.layers.Dense( 128, activation='relu' )( model_pre.output )
        x = tf.keras.layers.Dense( 128, activation='relu' )(x)
        outputs = tf.keras.layers.Dense( 131, activation='softmax' )(x)
        model   = tf.keras.Model( inputs=inputs, outputs=outputs )

        model.compile( optimizer='adam', 
                       loss='categorical_crossentropy', 
                       metrics = ['accuracy', Precision( name = 'precision' ), Recall( name = 'recall' ) ] )
        
        model.load_weights( self.model_weights )

        return model

    def file_upload( self ):
        st.write(f'\n\n\n\n')
        file = st.file_uploader( '', type=['jpeg', 'jpg'] )
        
        if file is None:
            st.title('Provide a Image :)')

        else:
            st.write(f'\n')
            st.image( file, use_column_width=True )
            st.write(f'\n')
            file = file.getvalue()
            file = tf.convert_to_tensor( file )

        return file

    def new_img( self, img_path ):
        image = tf.io.decode_jpeg( img_path )
        image = tf.image.resize( image, [224, 224], method='bilinear' )
        image = tf.expand_dims( image, 0 )

        return image

    def model_predict( self, image, model ):
        labels_u = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 2', 'Apple Golden 1', 'Apple Golden 3', 'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger','Banana Red', 'Beetroot', 'Blueberry', 'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Corn Husk', 'Cucumber Ripe', 'Cucumber Ripe 2', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes','Lychee','Mandarine','Mango','Mango Red','Mangostan','Maracuja','Melon Piel de Sapo','Mulberry','Nectarine','Nectarine Flat','Nut Forest','Nut Pecan','Onion Red','Onion Red Peeled','Onion White','Orange','Papaya','Passion Fruit','Peach','Peach 2','Peach Flat','Pear','Pear 2','Pear Abate','Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pepino', 'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon', 'Tomato Yellow', 'Tomato not Ripened', 'Walnut', 'Watermelon']

        prev = model.predict( image )

        percentages = []
        for i in range( 0, 131, 1 ):
            percentages.append( prev[0][i] )

        p_dict = { 'Labels': labels_u, 'Percentages': percentages }
        dataframe = DataFrame( p_dict )
        dataframe = dataframe[dataframe['Percentages'] > .10]

        if dataframe.empty:
            st.write('I dont Know what is that ;-;')

        return dataframe

if __name__ == '__main__':
    fruits = Fruits()
    fruits.pre_html()
 
    file = fruits.file_upload()

    if file is not None:
        image = fruits.new_img( img_path=file )

        model = fruits.model_preparation()
        
        z = fruits.model_predict( image=image, model=model )

        st.write( z )

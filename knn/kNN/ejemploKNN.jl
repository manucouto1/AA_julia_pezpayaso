
using ScikitLearn
using Statistics

@sk_import neighbors: KNeighborsClassifier

using Random
using DelimitedFiles

include("funcionesUtiles.jl");

porcentajeTest = 0.2;
numEjecuciones = 50;

# Cargamos la base de datos de iris
iris = readdlm("iris.txt");

# Las entradas van a ser el tercer y cuarto atributo
entradas = iris[:,3:4];
# Cambiamos el tipo de la variable a un array de valores reales
entradas = convert(Array{Float64}, entradas);

# La salida deseada va a ser el quinto atributo
salidasDeseadas = iris[:,5];
# Cambiamos el tipo de la variable a un array de strings
salidasDeseadas = convert(Array{String}, salidasDeseadas);

# El kNN es un clasificador multiclase, no es necesario aplicar la estrategia "uno contra todos"

# Para poder usar distintas clases en esta implementacion del kNN, cada una debe de estar representada por un numero distinto
# A esto se llama "one cold encoding", se puede hacer de la siguiente manera:
salidasDeseadas = indexin( salidasDeseadas, unique(salidasDeseadas) );


# Cuantos patrones tenemos
numPatrones = size(entradas, 1);

# Vamos a normalizar las entradas entre un maximo y un minimo
# En este caso, cada patron esta en una fila (al trabajar con redes de neuronas esto es al reves)
valoresNormalizacion = normalizarMaxMin!(entradas; patronesEnFilas=true);

# Dado que es un problema de clasificacion, no hay que normalizar la salida
# En un problema de regresion si que habria que normalizarla

# Creamos un par de vectores vacios donde vamos a ir guardando los resultados de cada ejecucion
precisionesEntrenamiento = Array{Float64,1}();
precisionesTest = Array{Float64,1}();

# Hacemos varias ejecuciones
for numEjecucion in 1:numEjecuciones

    println("Ejecucion ", numEjecucion);

    # Seleccionamos los patrones de entrenamiento y test: creamos indices
    # Esto es hacer un hold out
    (indicesPatronesEntrenamiento, indicesPatronesTest) = holdOut(numPatrones, porcentajeTest);

    # Creamos el kNN
    modelo = KNeighborsClassifier(3);

    # Ajustamos el modelo
    fit!(modelo, entradas[indicesPatronesEntrenamiento,:], salidasDeseadas[indicesPatronesEntrenamiento]);

    # Calculamos la clasificacion de los patrones de entrenamiento
    clasificacionEntrenamiento = predict(modelo, entradas[indicesPatronesEntrenamiento,:]);
    # Calculamos la precision en el conjunto de entrenamiento y la mostramos por pantalla
    precisionEntrenamiento = 100*mean(clasificacionEntrenamiento .== salidasDeseadas[indicesPatronesEntrenamiento]);
    println("Precision en el conjunto de entrenamiento: $precisionEntrenamiento %");

    # Calculamos la clasificacion de los patrones de test
    clasificacionTest = predict(modelo, entradas[indicesPatronesTest,:]);
    # Calculamos la precision en el conjunto de test y la mostramos por pantalla
    precisionTest = 100*mean(clasificacionTest .== salidasDeseadas[indicesPatronesTest]);
    println("Precision en el conjunto de test: $precisionTest %");

    # Y guardamos esos valores de precision obtenidos en esta ejecucion
    push!(precisionesEntrenamiento, precisionEntrenamiento);
    push!(precisionesTest, precisionTest);

end;

println("Resultados en promedio al separar las 3 clases:")
println("   Entrenamiento: ", mean(precisionesEntrenamiento), " %, desviacion tipica: ", std(precisionesEntrenamiento));
println("   Test:          ", mean(precisionesTest), " %, desviacion tipica: ", std(precisionesTest));

########################################################################################################################
# Si se quiere tener mas informacion del modelo:

# # Para ver los campos del modelo
# println(keys(modelo));
#
# # Por ejemplo:
# modelo.n_neighbors
# modelo.metric
# modelo.weights

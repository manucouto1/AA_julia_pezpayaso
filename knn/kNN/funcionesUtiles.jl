
using Statistics
using Random


#####################################################################################
#
# Para dividir los patrones en conjuntos de entrenamiento/test o entrenamiento/validacion/test
#
function holdOut(numPatrones::Int, porcentajeTest::Float64)
    @assert ((porcentajeTest>=0.) & (porcentajeTest<=1.));
    indices = randperm(numPatrones);
    numPatronesEntrenamiento = Int(round(numPatrones*(1-porcentajeTest)));
    return (indices[1:numPatronesEntrenamiento], indices[numPatronesEntrenamiento+1:end]);
end

function holdOut(numPatrones::Int, porcentajeValidacion::Float64, porcentajeTest::Float64)
    @assert ((porcentajeValidacion>=0.) & (porcentajeValidacion<=1.));
    @assert ((porcentajeTest>=0.) & (porcentajeTest<=1.));
    @assert ((porcentajeValidacion+porcentajeTest)<=1.);
    indices = randperm(numPatrones);
    numPatronesValidacion = Int(round(numPatrones*porcentajeValidacion));
    numPatronesTest = Int(round(numPatrones*porcentajeTest));
    numPatronesEntrenamiento = numPatrones - numPatronesValidacion - numPatronesTest;
    return (indices[1:numPatronesEntrenamiento], indices[numPatronesEntrenamiento+1:numPatronesEntrenamiento+numPatronesValidacion], indices[numPatronesEntrenamiento+numPatronesValidacion+1:numPatronesEntrenamiento+numPatronesValidacion+numPatronesTest]);
end




#####################################################################################
#
# Para normalizar
#
# patronesEnFilas=true para SVM, kNN, etc.
# patronesEnFilas=false para RNA
#
function normalizarMediaDesv!(patrones::Array{Float64,2}; patronesEnFilas=false)
    media      = mean(patrones, dims=(patronesEnFilas ? 1 : 2));
    desvTipica =  std(patrones, dims=(patronesEnFilas ? 1 : 2));
    patrones .-= media;
    patrones ./= desvTipica;
    if (patronesEnFilas)
        patrones[:, vec(desvTipica.==0)] .= 0;
    else
        patrones[vec(desvTipica.==0), :] .= 0;
    end
    return (patronesEnFilas, media, desvTipica)
end;

function normalizarMediaDesv(patrones::Array{Float64,2}; patronesEnFilas=false)
    patrones2 = copy(patrones);
    valoresNormalizacion = normalizarMediaDesv!(patrones2; patronesEnFilas=patronesEnFilas);
    return (patrones2, valoresNormalizacion);
end;

function normalizarMaxMin!(patrones::Array{Float64,2}; patronesEnFilas=false)
    minimo = minimum(patrones, dims=(patronesEnFilas ? 1 : 2));
    maximo = maximum(patrones, dims=(patronesEnFilas ? 1 : 2));
    patrones .-= minimo;
    patrones ./= (maximo .- minimo);
    if (patronesEnFilas)
        patrones[:, vec(minimo.==maximo)] .= 0;
    else
        patrones[vec(minimo.==maximo), :] .= 0;
    end
    return (patronesEnFilas, minimo, maximo)
end;

function normalizarMaxMin(patrones::Array{Float64,2}; patronesEnFilas=false)
    patrones2 = copy(patrones);
    valoresNormalizacion = normalizarMaxMin!(patrones2; patronesEnFilas=patronesEnFilas);
    return (patrones2, valoresNormalizacion);
end;

function desnormalizarMaxMin!(datos, valoresNormalizacion)
    @assert (ndims(datos)==2);
    (patronesEnFilas, minimo, maximo) = valoresNormalizacion;
    @assert (length(minimo) == size(datos, patronesEnFilas ? 2 : 1)) "No coincide el numero de atributos";
    @assert (length(maximo) == size(datos, patronesEnFilas ? 2 : 1)) "No coincide el numero de atributos";
    datos .*= (maximo .- minimo);
    datos .+= minimo;
    return nothing;
end;

function desnormalizarMaxMin(datos, valoresNormalizacion)
    datos2 = copy(datos);
    desnormalizarMaxMin!(datos2, valoresNormalizacion);
    return datos2;
end;

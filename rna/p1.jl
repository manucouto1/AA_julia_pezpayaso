using FileIO
using Images
using Statistics
using Flux: onehotbatch, onecold, crossentropy, binarycrossentropy
using Flux
using Random
using DelimitedFiles
using ImageView, Images

include("funcionesUtiles.jl");

function train_model(numEntradasRNA, numSalidasRNA, ArquitecturaRNA, patrones, salidasDeseadas)

    PorcentajeValidacion = 0.2;
    PorcentajeTest = 0.2;

    NumMaxCiclosEntrenamiento = 1000;
    NumMaxCiclosSinMejorarValidacion = 100;

    numEjecuciones = 50;
    numPatrones = size(patrones,2);

    precisionesEntrenamiento = Array{Float64,1}();
    precisionesValidacion = Array{Float64,1}();
    precisionesTest = Array{Float64,1}();

    @assert (size(patrones,2)==size(salidasDeseadas,2));
    # Si vamos a separar dos clases, solamente necesitamos una salida
    #  Por tanto, nos aseguramos de que no haya 2 salidas
    @assert (numSalidasRNA!=2);

    mejorModelo = Chain(
            Dense(numEntradasRNA, ArquitecturaRNA[1], σ),
            Dense(ArquitecturaRNA[1], numSalidasRNA, σ), );

    for numEjecucion in 1:numEjecuciones
        print("Ejecucion ", numEjecucion)
        (indicesPatronesEntrenamiento, indicesPatronesValidacion, indicesPatronesTest) = 
            holdOut(numPatrones, PorcentajeValidacion, PorcentajeTest);
        
        funcionTransferenciaCapasOcultas = σ; 
        
        if(length(ArquitecturaRNA) == 1) 
            modelo = Chain(
                Dense(numEntradasRNA,     ArquitecturaRNA[1], funcionTransferenciaCapasOcultas),
                Dense(ArquitecturaRNA[1], numSalidasRNA,      σ                               ), );
        elseif(length(ArquitecturaRNA)==2)
            modelo = Chain(
                Dense(numEntradasRNA,     ArquitecturaRNA[1], funcionTransferenciaCapasOcultas),
                Dense(ArquitecturaRNA[1], ArquitecturaRNA[2], funcionTransferenciaCapasOcultas),
                Dense(ArquitecturaRNA[2], numSalidasRNA,      σ                               ), );
        else 
            error("Para redes MLP, no hacer más de 2 capas ocultas")
        end
        println(modelo)
        
        loss(x, y) = (size(y,1)>1) ? 
            crossentropy(modelo(x), y) : 
            mean(binarycrossentropy.(modelo(x), y));
        
        # Entrenamos la RNA hasta que se cumpla el criterio de parada
        criterioFin = false;
        numCiclo = 0; mejorLossValidacion = Inf; numCiclosSinMejorarValidacion = 0; mejorModelo = nothing;
        while (!criterioFin)
    
            Flux.train!(loss, params(modelo), [(patrones[:,indicesPatronesEntrenamiento], salidasDeseadas[:,indicesPatronesEntrenamiento])], ADAM(0.01));
    
            numCiclo += 1;
    
            if (PorcentajeValidacion>0)
                lossValidacion = loss(patrones[:,indicesPatronesValidacion], salidasDeseadas[:,indicesPatronesValidacion]);
                if (lossValidacion<mejorLossValidacion)
                    mejorLossValidacion = lossValidacion;
                    mejorModelo = deepcopy(modelo);
                    numCiclosSinMejorarValidacion = 0;
                else
                    numCiclosSinMejorarValidacion += 1;
                end;
            end;
    
            if (numCiclo>=NumMaxCiclosEntrenamiento)
                criterioFin = true;
            end;
            if (numCiclosSinMejorarValidacion>NumMaxCiclosSinMejorarValidacion)
                criterioFin = true;
            end;
    
        end;
        
        if (PorcentajeValidacion>0)
            modelo = mejorModelo;
        end;
        # Fin del entrenamiento de la RNA
        
        println("   RNA entrenada durante ", numCiclo, " ciclos");
         precision(x, y) = (size(y,1)==1) ? 
            mean((modelo(x).>=0.5) .== y) : 
            mean(onecold(modelo(x)) .== onecold(y));
        
        precisionEntrenamiento = 100*precision(patrones[:,indicesPatronesEntrenamiento], salidasDeseadas[:,indicesPatronesEntrenamiento]);
        println("   Precision en el conjunto de entrenamiento: $precisionEntrenamiento %");
        push!(precisionesEntrenamiento, precisionEntrenamiento);
    
        if (PorcentajeValidacion>0)
            precisionValidacion = 100*precision(patrones[:,indicesPatronesValidacion], salidasDeseadas[:,indicesPatronesValidacion]);
            println("   Precision en el conjunto de validacion: $precisionValidacion %");
            push!(precisionesValidacion,    precisionValidacion);
        end;
    
        precisionTest = 100*precision(patrones[:,indicesPatronesTest], salidasDeseadas[:,indicesPatronesTest]);
        println("   Precision en el conjunto de test: $precisionTest %");
        push!(precisionesTest,          precisionTest);
        
    end;

    println("Resultados en promedio:");
    println("Entrenamiento: ", mean(precisionesEntrenamiento), " %, desviacion tipica: ", std(precisionesEntrenamiento));
    if (PorcentajeValidacion>0)
        println("Validacion: ", mean(precisionesValidacion)," %, desviacion tipica: ", std(precisionesValidacion));
    end;
    println("Test: ", mean(precisionesTest), " %, desviacion tipica: ", std(precisionesTest));

    return mejorModelo;
end;

function get_orange_white_boolean(file)
    imagen = load(file)
    diferenciaRojoVerde = 0.3; diferenciaRojoAzul = 0.3;
    canalRojo = red.(imagen); 
    canalVerde = green.(imagen); 
    canalAzul = blue.(imagen);
    white_chanel = (zeros(size(canalRojo)[1], size(canalRojo)[2]).+0.8);
    
    mb_w =  (canalRojo.>white_chanel) .& 
            (canalVerde.>white_chanel) .& 
            (canalAzul.>white_chanel);
    
    mb_r =  (canalRojo.>(canalVerde.+diferenciaRojoVerde)) .& 
            (canalRojo.>(canalAzul.+diferenciaRojoAzul));
    
    matrizBooleana = mb_w .+ mb_r;
    
    return matrizBooleana;
end

function check_region(x1, y1, x2, y2, imagen, modelo, nImputs, wall)
    img_v = @view imagen[x1:x2, y1:y2]
    img_CHW = channelview(img_v)

    xtrain = process_image(img_v, nImputs);
    
    if(modelo(reshape(xtrain,nImputs,1))[1,1] > wall)
        imagen[x1:x2, y1] .= RGB(0,1,0);
        imagen[x1:x2, y2] .= RGB(0,1,0);
        imagen[x1, y1:y2] .= RGB(0,1,0);
        imagen[x2, y1:y2] .= RGB(0,1,0);
    else
        imagen[x1:x2, y1] .= RGB(1,0,0);
        imagen[x1:x2, y2] .= RGB(1,0,0);
        imagen[x1, y1:y2] .= RGB(1,0,0);
        imagen[x2, y1:y2] .= RGB(1,0,0);
    end
end

function process_picture(filePath, modelo, nImputs, wall)
    booleanMatrix = get_orange_white_boolean(filePath);
    labelArray = ImageMorphology.label_components(booleanMatrix);
    boundingBoxes = ImageMorphology.component_boxes(labelArray);
    tamanos = ImageMorphology.component_lengths(labelArray); #Tamaño del grupo 
    pixeles = ImageMorphology.component_indices(labelArray);
    pixeles = ImageMorphology.component_subscripts(labelArray);
    centroides = ImageMorphology.component_centroids(labelArray);

    tamanos = component_lengths(labelArray);
    etiquetasEliminar = findall(tamanos .<= 370) .- 1;
    matrizBooleana = [!in(etiqueta,etiquetasEliminar) && (etiqueta!=0) for etiqueta in labelArray];
    
    labelArray = ImageMorphology.label_components(matrizBooleana);
    imagenObjetos = RGB.(matrizBooleana, matrizBooleana, matrizBooleana);
    centroides = ImageMorphology.component_centroids(labelArray)[2:end];
    for centroide in centroides
        x = Int(round(centroide[1]));
        y = Int(round(centroide[2]));
        imagenObjetos[ x, y ] = RGB(1,0,0);
    end;
    # Vamos a recuadrar el bounding box de estos objetos, en color verde
    # Calculamos los bounding boxes, y eliminamos el primero (el objeto "0")
    imagen = load(filePath)
    
    boundingBoxes = ImageMorphology.component_boxes(labelArray)[2:end];
    for boundingBox in boundingBoxes
        x1 = boundingBox[1][1];
        y1 = boundingBox[1][2];
        x2 = boundingBox[2][1];
        y2 = boundingBox[2][2];

        check_region(x1,y1,x2,y2, imagen, modelo, nImputs, wall)
    end;
    display(imagen);
end;

function process_image(image, n_inputs)
    img_CHW = channelview(image);
    patron = zeros(n_inputs);
    
    patron[1] = mean(img_CHW[1,:,:]);
    patron[2] = mean(img_CHW[2,:,:]);
    patron[3] = mean(img_CHW[3,:,:]);
    patron[4] = std(img_CHW[1,:,:]);
    patron[5] = std(img_CHW[2,:,:]);
    patron[6] = std(img_CHW[3,:,:]);

    return reshape(patron,n_inputs,1);
end;

function main()

    patrones_si, salidas_deseadas_si = prepare_data("positivos", 1);
    patrones_no, salidas_deseadas_no = prepare_data("negativos", 0);
    
    patrones = hcat(patrones_si, patrones_no);
    salidasDeseadas = hcat(salidas_deseadas_si, salidas_deseadas_no);

    mejorModelo = train_model(6,1,[5, 4], patrones, salidasDeseadas);

    directorio = readdir("fotos");

    for file in directorio 
        println(file);
        process_picture(string("fotos/",file), mejorModelo, 6, 0.72);
    end;

end



main()
using FileIO
using Images
using Statistics
using Flux: onehotbatch, onecold, crossentropy, binarycrossentropy
using Flux
using Random
using DelimitedFiles
using ImageView, Images, ImageSegmentation

include("../funcionesUtiles.jl");

function get_main_color(img, qValue1, qValue2, filter)
  
    matrizBooleana = filter.(img);
    matrizBooleana = closing(matrizBooleana)
    
    labelArray = ImageMorphology.label_components(matrizBooleana);
    
    boundingBoxes = ImageMorphology.component_boxes(labelArray);
    tamanos = ImageMorphology.component_lengths(labelArray); #Tamaño del grupo 
    pixeles = ImageMorphology.component_indices(labelArray);
    pixeles = ImageMorphology.component_subscripts(labelArray);
    centroides = ImageMorphology.component_centroids(labelArray);
    
    min_size = (size(img)[1]*size(img)[2]/600);
    
    etiquetasEliminar = findall(tamanos .<= min_size) .- 1; # Importate el -1, porque la primera etiqueta es la 0

    matrizBooleana = [!in(etiqueta,etiquetasEliminar) && (etiqueta!=0) for etiqueta in labelArray];
    
    labelArray = ImageMorphology.label_components(matrizBooleana);
    
    centroides = ImageMorphology.component_centroids(labelArray)[2:end];

    boundingBoxes = ImageMorphology.component_boxes(labelArray)[2:end];
    
    i = 1;
    R1 = Array{Float64,1}(undef, size(boundingBoxes));
    G1 = Array{Float64,1}(undef, size(boundingBoxes));
    B1 = Array{Float64,1}(undef, size(boundingBoxes));
    
    img_CHW = channelview(img);
    
    i = 1;
    for boundingBox in boundingBoxes
        x1 = boundingBox[1][1];
        y1 = boundingBox[1][2];
        x2 = boundingBox[2][1];
        y2 = boundingBox[2][2];

        R1[i] = quantile(skipmissing(img_CHW[1,x1:x2,y1:y2]), qValue1);
        G1[i] = quantile(skipmissing(img_CHW[2,x1:x2,y1:y2]), qValue1);
        B1[i] = quantile(skipmissing(img_CHW[3,x1:x2,y1:y2]), qValue1);
        i= i + 1;
    end;
    
    if(maximum(labelArray) == 0)
        return (quantile(skipmissing(red.(img)), qValue2), quantile(skipmissing(green.(img)), qValue2), quantile(skipmissing(blue.(img)),qValue2))
    else 
        return (quantile(skipmissing(R1), qValue2), quantile(skipmissing(G1), qValue2), quantile(skipmissing(B1),qValue2))
    end
end

esPixelRojo(pixel::RGBA, dRV=0.3, dRA=0.3) = (pixel.r > pixel.g + dRV) && (pixel.r > pixel.b + dRA);
esPixelRojo(pixel::RGB, dRV=0.3, dRA=0.3) = (pixel.r > pixel.g + dRV) && (pixel.r > pixel.b + dRA);
esPixelBlanco(pixel::RGB, dr=0.3, dg=0.3, db=0.3) = (pixel.r + dr >= 1) && (pixel.g + dg >= 1) && (pixel.b + db >= 1);
esPixelBlanco(pixel::RGBA, dr=0.3, dg=0.3, db=0.3) = (pixel.r + dr >= 1) && (pixel.g + dg >= 1) && (pixel.b + db >= 1);

function process_image(image)
    img = copy(image);
    img_CHW = channelview(img);
    
    (R1, G1, B1) = get_main_color(img, 0.99, 0.99, esPixelRojo)
    (R2, G2, B2) = get_main_color(img, 0.75, 0.95, esPixelBlanco)

    patron = [
        mean(img_CHW[1,:,:]), 
        mean(img_CHW[2,:,:]), 
        mean(img_CHW[3,:,:]),
        std(img_CHW[1,:,:]), 
        std(img_CHW[2,:,:]), 
        std(img_CHW[3,:,:]),
        R1,
        G1,
        B1,
        R2,
        G2,
        B2];
        
    return patron;
end;

function prepare_data(dir, n_inputs, output, patronesEnFilas = true)
    directorio = readdir(dir);
    
    salidas_deseadas = zeros(1, size(directorio,1));
    patrones = Array{Float64,2}(undef, size(directorio,1),n_inputs);
    
    i = 1
    
    if(patronesEnFilas)
        for file in directorio 
            image = load(string(dir,"/",file));
            patrones[i,:] = process_image(image)[:];
            salidas_deseadas[i] = output;
            i = i + 1
            
        end;
    else
        for file in directorio 
            image = load(string(dir,"/",file));
            patrones[:,i] = process_image(image)[:];
            salidas_deseadas[i] = output;
            i = i + 1
        end;
    end

    return patrones, salidas_deseadas;
end;

function get_orange_white_boolean(imagen)
    
    diferenciaRojoVerde = 0.3; 
    diferenciaRojoAzul = 0.3;
    
    canalRojo = red.(imagen); 
    canalVerde = green.(imagen); 
    canalAzul = blue.(imagen);
    
    white_chanel = (zeros(size(canalRojo)[1], size(canalRojo)[2]).+0.6);
    white_chanel1 = (zeros(size(canalRojo)[1], size(canalRojo)[2]).+0.9);
    
    mb_w =  (canalRojo.>white_chanel.+ mean(canalRojo)) .& 
            (canalVerde.>white_chanel.- mean(canalVerde)) .& 
            (canalAzul.>white_chanel.- mean(canalAzul));
    
    mb_w1 =  (canalRojo.>white_chanel1) .& 
            (canalVerde.>white_chanel1) .& 
            (canalAzul.>white_chanel1);
    
    mb_r =  (canalRojo.>(canalVerde.+diferenciaRojoVerde)) .& 
            (canalRojo.>(canalAzul.+diferenciaRojoAzul));
    
    matrizBooleana = mb_r .| mb_w1 #.| mb_w1; 
    
    return matrizBooleana;
end;

function check_region(x1, y1, x2, y2, imagen, modelo, nImputs, wall)
    img_v = @view imagen[x1:x2, y1:y2]
    img_CHW = channelview(img_v)

    xtrain = convert(Array{Float64}, process_image(img_v));
    
    xtrain_aux =  reshape(xtrain,1,nImputs);
    if(modelo.predict(xtrain_aux)[1] == 1.0)
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

function train_model(numEntradasRNA, numSalidasRNA, ArquitecturaRNA, patrones, salidasDeseadas)

    PorcentajeValidacion = 0.2;
    PorcentajeTest = 0.2;

    NumMaxCiclosEntrenamiento = 1000;
    NumMaxCiclosSinMejorarValidacion = 100;

    numEjecuciones = 10;
    numPatrones = size(patrones,2);

    precisionesEntrenamiento = Array{Float64,1}();
    precisionesValidacion = Array{Float64,1}();
    precisionesTest = Array{Float64,1}();
    
    

    @assert (size(patrones,2)==size(salidasDeseadas,2));
    # Si vamos a separar dos clases, solamente necesitamos una salida
    #  Por tanto, nos aseguramos de que no haya 2 salidas
    @assert (numSalidasRNA!=2);
    
    
    
    mmArq = Chain(
        Dense(numEntradasRNA, ArquitecturaRNA[1], σ),
        Dense(ArquitecturaRNA[1], numSalidasRNA, σ), );
    
    mmPrecisionTest=0;
    mmC1 = 1
    mmC2 = 1
    
    mejorCapa1 = 1
    mejorCapa2 = 1
    
    for capa1 in 1:ArquitecturaRNA[1]
        for capa2 in 1:ArquitecturaRNA[2]
            mejorModelo = Chain(
                Dense(numEntradasRNA, ArquitecturaRNA[1], σ),
                Dense(ArquitecturaRNA[1], numSalidasRNA, σ), );
            mejorLossValidacion = 0
            println("Arquitectura : ",numEntradasRNA, ":",capa1,":",capa2,":1")
            
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
                        Dense(numEntradasRNA,     capa1, funcionTransferenciaCapasOcultas),
                        Dense(capa1, capa2, funcionTransferenciaCapasOcultas),
                        Dense(capa2, numSalidasRNA,      σ                               ), );
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
                            mejorCapa1 = capa1
                            mejorCapa2 = capa2
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
            if (PorcentajeValidacion > 0)
                println("Validacion: ", mean(precisionesValidacion)," %, desviacion tipica: ", std(precisionesValidacion));
            end;
            println("Test: ", mean(precisionesTest), " %, desviacion tipica: ", std(precisionesTest));
            
            ## TODO - Comprobar que la media de precisiones de entrenamiento de la aquitectura sea la adecuada
            mediaAux = mean(precisionesTest)[1]
            println(mean(precisionesTest))
            
            if(mediaAux > mmPrecisionTest)
                mmArq = mejorModelo
                mmPrecisionTest = mediaAux
                mmC1 = mejorCapa1
                mmC2 = mejorCapa2
                println("Cambio de Mejor : ",12,":",mmC1,":",mmC2);
            end
        end
    end
    return mmArq, mmC1, mmC2;
end;

function check_region(x1, y1, x2, y2, imagen, modelo, nImputs, wall)
    img_v = @view imagen[x1:x2, y1:y2]
    img_CHW = channelview(img_v)

    xtrain = convert(Array{Float64}, process_image(img_v));
    
    xtrain_aux =  reshape(xtrain,1,nImputs);
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
    imagen = load(filePath);
    booleanMatrix = get_orange_white_boolean(imagen)
    booleanMatrix = closing(dilate(booleanMatrix,[1,2]))

    if(!maximum(booleanMatrix))
        booleanMatrix = get_orange_white_boolean(imagen);
    end;
    
    labelArray = ImageMorphology.label_components(booleanMatrix);
    boundingBoxes = ImageMorphology.component_boxes(labelArray);
    tamanos = ImageMorphology.component_lengths(labelArray); #Tamaño del grupo 
    pixeles = ImageMorphology.component_indices(labelArray);
    pixeles = ImageMorphology.component_subscripts(labelArray);
    centroides = ImageMorphology.component_centroids(labelArray);

    min_size = (size(imagen)[1]*size(imagen)[2]/600);
    etiquetasEliminar = findall(tamanos .<= min_size) .- 1;
    matrizBooleana = [!in(etiqueta,etiquetasEliminar) && (etiqueta!=0) for etiqueta in labelArray];
    
    labelArray = ImageMorphology.label_components(matrizBooleana);
    imagenObjetos = RGB.(matrizBooleana, matrizBooleana, matrizBooleana);
    centroides = ImageMorphology.component_centroids(labelArray)[2:end];
    
    for centroide in centroides
        x = Int(round(centroide[1]));
        y = Int(round(centroide[2]));
        imagenObjetos[ x, y ] = RGB(1,0,0);
    end;
    
    imagen = load(filePath)
    
    boundingBoxes = ImageMorphology.component_boxes(labelArray)[2:end];
    
    for boundingBox in boundingBoxes
        x1 = boundingBox[1][1];
        y1 = boundingBox[1][2];
        x2 = boundingBox[2][1];
        y2 = boundingBox[2][2];

        check_region(x1,y1,x2,y2, imagen, modelo, nImputs, wall)
    end;
    return imagen;
end;

patrones_si, salidas_deseadas_si = prepare_data("../db/positivos", 12, 1);
patrones_no, salidas_deseadas_no = prepare_data("../db/negativos", 12, 0);

patrones = vcat(patrones_si, patrones_no);
salidasDeseadas = hcat(salidas_deseadas_si, salidas_deseadas_no);
patrones = permutedims(patrones,[2,1]);

mejorModelo, mejorC1, mejorC2 = train_model(12,1,[12, 10], patrones, salidasDeseadas);

mejorModelo

directorio = readdir("../db/fotos");

for file in directorio 
    println(file);
    imagen = process_picture(string("../db/fotos/",file), mejorModelo, 12, 0.55);
    display(imagen)
end; 


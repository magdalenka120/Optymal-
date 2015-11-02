# Optymal-%--------------------------------------------------------------------------
%-----------------------  Optymalizacja systemów  -------------------------
%--------------------------------------------------------------------------
% Zadanie 1: Detekcja twarzy
% autorzy: A. Gonczarek, J.M. Tomczak
% 2014
%--------------------------------------------------------------------------


function [L, grad] = logistic_cost_function( xTrain, yTrain, w )
% Funkcja wylicza wartosc funkcji celu L i jej gradient grad
% xTrain - ciag treningowy wejsciowy NxD
% yTrain - ciag zero-jedynkowy klas Nx1
% w - parametry modelu

N = size(xTrain,1);
M = size(w,1);

L = 0;
grad = zeros(M,1);

%--------------------------------------------------------------------------
%--------------------- TUTAJ WLASNA IMPLEMENTACJA -------------------------
%--------------------------------------------------------------------------

L = L - sum( yTrain .* log(sigmoid( xTrain*w)) ) - sum( (1-yTrain) .* log(1-sigmoid(xTrain*w)) );
grad = grad + xTrain' * (sigmoid(xTrain * w) - yTrain);






%--------------------------------------------------------------------------

end

%--------------------------------------------------------------------------
%-----------------------  Optymalizacja systemów  -------------------------
%--------------------------------------------------------------------------
% Zadanie 1: Detekcja twarzy
% autorzy: A. Gonczarek, J.M. Tomczak
% 2014
%--------------------------------------------------------------------------

function [w func_values] = gradient_descent( xTrain, yTrain, w0, eps, eta )
% Funkcja wykonuje optymalizacje metoda gradientu prostego dla funkcji
% logistic_cost_function z dokladnoscia eps i krokiem step. Zwraca rozwiazane w, ktore
% minimalizuje funkcje celu oraz wartosci funkcji we wszystkich krokach
% algorytmu - func_values.

% xTrain - dane treningowe wejsciowe
% yTrain - dane treningowe wyjsciowe (etykiety klas)
% w0 - punkt startowe (poczatkowe parametry)
% eps - dokladnosc dla warunku stopu
% eta - krok optymalizacji

func_values = [];
w = zeros(size(w0));
    
    
%--------------------------------------------------------------------------
%--------------------- TUTAJ WLASNA IMPLEMENTACJA -------------------------
%--------------------------------------------------------------------------

w = w0;
wd = ones(size(w0));
while norm(w-wd) >= eps,
    wd = w;
    [L, grad] = logistic_cost_function( xTrain, yTrain, w );
    func_values = [func_values ;L];
    w = w - grad*eta;
end








%--------------------------------------------------------------------------

end

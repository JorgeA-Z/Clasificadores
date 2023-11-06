import clasificador as cl

class MainUserInterface:

    def __init__(self) -> None:
        self.pima = cl.Clasificador(1)
        self.autos = cl.Clasificador(2)
        self.wine = cl.Clasificador(3)

        self.pima.read_Dataset()
        self.autos.read_Dataset()
        self.wine.read_Dataset()

    def Text(self):
        print("-------------------------")
        print("1. Regresión logística")
        print("2. K-Vecinos Cercanos")
        print("3. Maquinas Vector Soporte")
        print("4. Naive Bayes")
        print("5. Red Neuronal")
        print("6. Salir")
        print("-------------------------")

        
    def Menu(self):
        llave = True

        while llave:
            self.Text();

            try:
                op = int(input("Ingresa una opcion: "))

            except Exception as e:
                print(f"Error: {e}\n\n\n\n\n\n\n\n\n\n\n\n")

            else:
                if op == 1:
                    #Regresión logistica
                    self.pima.Logistic_Regression()
                    self.wine.Logistic_Regression()
                    #self.autos.Logistic_Regression()

                elif op == 2:
                    #K vecinos
                    self.pima.K_Nearest_Neighbors()
                    self.wine.K_Nearest_Neighbors()

                elif op == 3:
                    #Maquinas vector soporte
                    self.pima.Support_Vector_Machines()
                    self.wine.Support_Vector_Machines()

                elif op == 4:
                    #Naive Bayes
                    self.pima.Naive_Bayes()
                    self.wine.Naive_Bayes()

                elif op == 5:      
                    #Red Neuronal 
                    self.pima.redNeuronal()
                    self.wine.redNeuronal()

                elif op == 6:
                    
                    llave = False

                    print("Gracias por usar el programa")

                else:
                    print("Opcion invalida\n\n\n\n\n\n\n\n\n\n\n\n")

if __name__ == "__main__":
    menu = MainUserInterface()

    menu.Menu()

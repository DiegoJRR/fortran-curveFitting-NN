      !Producto integrador de aprendizaje.
      !M‚todo num‚rico de interpolaci¢n con una red neuronal.
      !M¢dulo de la red neuronal utilizado:
        module class_nerualNetwork
      !Declaraci¢n de variables:
	implicit none
	private
      !Declaraci¢n de la red neuralNetwork, subrutinas y funciones.
        public::nerualNetwork, init , FeedForward, loss, BackPropagate
        public::train
      !Declaracip¢n de la red neuronal y sus arreglos.
	type nerualNetwork
	 	integer::n
	real::eta
	real::b_2
	 	real::w_1(50), b_1(50), w_2(50)
	 	real::v_1(50), y_1(50), v_2
	 	real::r=0
	end type nerualNetwork
      !Contenido del m¢dulo, subrutinas, y funciones.
	contains
	subroutine init(this)
		type(nerualNetwork), intent(inout)::this
		integer::i=1
	call random_seed()
		call random_number(this%w_1)
		call random_number(this%w_2)

		i=1
		do while (i <= this%n)
			this%b_1(i) = 1
	           i = i + 1
		end do

	end subroutine

	function FeedForward(this, x) result(r)
		type(nerualNetwork), intent(inout)::this
		real::dotProduct
		real, intent(in)::x
		real::c(this%n)
		real::r
		integer::i=1

		call scalarMult(x, this%w_1, c, this%n)
		call linComb(1.0, c, 1.0, this%b_1, this%v_1, this%n)

		i=1
		do while(i<=this%n)
                        !this%y_1(i) = tanh(this%v_1(i))
			this%y_1(i) = tanh(this%v_1(i))
			i = i + 1
		end do

	this%v_2 = dotProduct(this%y_1, this%w_2, this%n) + this%b_2

		r = this%v_2
	end function

	function loss(this, x, d, n) result(cost)
		type(nerualNetwork), intent(inout)::this

		integer, intent(in)::n
		real, intent(in)::x(n), d(n)
		real::c(n)
	        real::temp
		integer::i=1
	        real::cost
	        i = 1
	        temp = 0
			do while(i.le.n)
				c(i) = d(i) - FeedForward(this, x(i))
				temp = temp + (c(i)**2)
				i = i + 1
			end do

			cost = temp/n
		end function

		subroutine BackPropagate(this, x, y, d)
			type(nerualNetwork), intent(inout)::this
	        real, intent(in)::x, y, d
			real::delta_out, t
			real::delta_1
	        dimension::delta_1(this%n)
			integer::i
			delta_out = d - y

			this%b_2 = this%b_2 + this%eta*delta_out

			i = 1
	do while(i<=this%n)
	this%w_2(i) = this%w_2(i) + (this%eta)*delta_out*(this%y_1(i))
	     !t = 1 - (tanh(this%v_1(i)))**2
	t = 1 - (tanh(this%v_1(i)))**2
	delta_1(i) = t*this%w_2(i)*delta_out
	i = i + 1
	end do

	i = 1
	do while(i<=this%n)
		this%w_1(i) = this%w_1(i) + (this%eta)*x*delta_1(i)
		this%b_1(i) = this%b_1(i) + (this%eta)*delta_1(i)
		i = i + 1
	end do
	end subroutine

	subroutine train(this, x, d, n,  epoch)
		type(nerualNetwork), intent(inout)::this
		integer::iter, i, j
		real::prediction
		integer, intent(in)::epoch, n
		real, intent(in)::x(n), d(n)
		j = 1
		iter = 0
		do while (iter.ne.epoch)
			i=1
			do while(i<=n)
			prediction = FeedForward(this, x(i))
			call BackPropagate(this, x(i), prediction, d(i))

			i = i + 1
		end do

	if(mod(iter, epoch/10).eq.0) then
        write(*,*) "poca:", iter, " p‚rdida:", loss(this, x, d, 300),&
	&" ", "[", repeat ('%', j), repeat("-", 10 - j), "]"
	j = j + 1
	end if

		iter = iter + 1
	end do
	end subroutine

	end module class_nerualNetwork

	program neural
	use class_nerualNetwork
	implicit none
	type(nerualNetwork)::nn
	character:: archi
	real::x(300), y(300), v(300), d(300), a, va, vb, resp
       real, allocatable::la(:), lb(:)
	integer::i, extra_epochs, valid, pr, r, mul10
	integer:: Reason, n_puntos
	logical:: file_exists

	call random_number(x)
	call random_number(y)
	call random_number(v)


	i = 1
	do while (i<=300)
	    !Para generar un numero random en el intervalo medio abierto [-1, 1)
	    !Esto despues se va a cambiar para que utilice los datos de entrada 
	    !leidos de un archivo csv
	  	!x(i) = (x(i)*2)-1
	  	v(i) = (v(i)*0.2) - 0.1
		d(i) = sin(20*x(i)) + 3*x(i) + v(i)
	    i = i + 1
	end do

	nn = nerualNetwork(50, 0.01, 1, 0, 0, 0, 0, 0, 0, 0)

      write(*,*) "Bienvenid@, presione enter para iniciar el programa."
	read(*,*)
      write(*,*)"======================================================"
      write(*,*)"Este programa usa un m‚todo num‚rico de interpolaci¢n"
      write(*,*)"para definir con una red neuronal, la cual interpola"
      write(*,*)"los datos obtenidos de una tabla dada por el usuario"
      write(*,*)"donde ingresa los puntos deseados a interporalizar."
      write(*,*)"======================================================"
      write(*,*)""
      write(*,*)"======================================================"
      write(*,*)"Para tener mejores reultados utilice m s de 50 puntos."
      write(*,*)"El programa  tiene la opcion de leer un archivo '.txt'"
      write(*,*)"que contenta los  puntos o pueden  ser  ingresados  de"
      write(*,*)"manera manual. Seleccione la fuente de los datos:"
      write(*,*)"======================================================"
      write(*,*)"-----------------Manual[1]/Archivo[2]-----------------"

      valid=0
      do while (valid.eq.0)
      read(*,*,IOSTAT=Reason) resp
      if (Reason.eq.0.and.(resp.eq.1.or.resp.eq.2)) then
      valid=1
      else
      write(*,*)"======================================================"
      write(*,*)"        Error: Ingrese una opci¢n v lida."
      write(*,*)"======================================================"
      end if
      end do
      
      
      if(resp.eq.1) then
         Write(*,*) "¨Cuantos puntos desea interpolar?"

         valid=0
         do while (valid.eq.0)
           read(*,*,IOSTAT=Reason) pr
           if (Reason.eq.0.and.pr.gt.0)  then
           valid=1
           else
           write(*,*)"================================================="
           write(*,*)"Error: Ingrese una cantidad de puntos num‚rica"
           write(*,*)"          sin decimales y mayor a 0."
           write(*,*)"================================================="
           end if
         end do
         
         allocate(la(pr), lb(pr))
         do i=1,pr
         
         write(*,*)"Ingrese la ordenada X del punto:",i

         valid=0
         do while (valid.eq.0)
         read(*,*,IOSTAT=Reason) la(i)
         if (Reason.eq.0) then
         valid=1
         else
         write(*,*)"==================================================="
         write(*,*)"        Error: Ingrese un valor nume‚rico."
         write(*,*)"==================================================="
         end if
         end do
         
         write(*,*)"Ingrese la ordenada Y del punto:", i
         
         valid=0
         do while (valid.eq.0)
         read(*,*,IOSTAT=Reason) lb(i)
         if (Reason.eq.0) then
         valid=1
         else
         write(*,*)"==================================================="
         write(*,*)"        Error: Ingrese un valor nume‚rico."
         write(*,*)"==================================================="
         end if
         end do
         
         end do
         
      else if (resp.eq.2) then
           write(*,*)"Ingrese sus datos dentro de un archivo nombrado"
           write(*,*)"datos.txt en el mismo directorio donde est "
           write(*,*) "ejecutando este programa."
           write(*,*)
           write(*,*) "Ingrese el numero de puntos en su archivo"
           
           read(*,*) n_puntos
           
           write(*,*) "Presione cualquier tecla cuando su archivo"
           write(*,*) "tenga todos los", n_puntos, "ingresados y se"
           write(*,*) "encuentre en el directorio correcto."
           

           !inquire(file=archi, exist=file_exists)
           !write(*,*) file_exists
           !if (file_exists.eqv..true.)then
           !valid=1
           !else
      !write(*,*)"==================================================="
      !write(*,*)"Error: No se encontr¢ el archivo vuelva a ingresarlo."
      !write(*,*)"==================================================="
           !end if
           !end do
      end if


      open(unit=1, file="datos.txt", status='old',&
      &access='sequential', action='read')

      write(*,*) "Leyendo los puntos del archivo "
      do i=1, n_puntos
         read(1, *) x(i), d(i)
         !write(*,*) c(i), d(i)
      end do

      close(1)
      
      write(*,*)""
      write(*,*) "Llamando a la red neuronal."
      write(*,*)""
      call init(nn)
      write(*,*) "P‚rdida inicial:", loss(nn, x, d, n_puntos)
	!write(*,*) "Network initialized with the following values:"
	!write(*,*) "n:", nn%n
	!write(*,*) "eta:", nn%eta
	!write(*,*) "b_2:", nn%b_2
	!write(*,*) "w_1:", nn%w_1
	!write(*,*) "w_2:", nn%w_2
	!write(*,*) "b_1:", nn%b_1
	!read(*,*)

      write(*,*) "Iniciando el entrenamiento... *beep beep boop*"
      call train(nn, x, d, n_puntos, 5000)

      write(*,*) "Red entrenada!"
      write(*,*) "P‚rdida final:", loss(nn, x, d, n_puntos)
      write(*,*)""
      write(*,*)"======================================================"
      write(*,*)"Para conocer los resultados graficamente utilice un"
      write(*,*)"programa de confianza como Excel, Gnuplot, etc."
      write(*,*)"Este programa genera un archivo.csv llamado:"
      write(*,*)"'predictions.csv'"
      write(*,*)"======================================================"
      write(*,*)""
      write(*,*)"======================================================"
      write(*,*)"Ingrese 1 para entrenar m s su red neuronal"
      write(*,*)"o 0 para cerrar el programa."
      write(*,*)"======================================================"
      
      valid=0
      do while (valid.eq.0)
      read(*,*) r
      if (r.eq.0.or.r.eq.1) then
      	valid=1
      else
      	write(*,*)"======================================================"
      	write(*,*)"        Error: Ingrese una respuesta valida."
      	write(*,*)"======================================================"
      	write(*,*) ""
	  	write(*,*)"======================================================"
      	write(*,*)"Ingrese 1 para entrenar mas su red neuronal"
      	write(*,*)"o 0 para cerrar el programa."
      	write(*,*)"======================================================"
      
      	read(*,*) r
      end if
      end do

	do while (r.eq.1)
		write(*,*) "Ingrese el n£mero de ‚pocas"
                write(*,*) "extras para entrenar la red neuronal."

                valid=0
                do while (valid.eq.0)
                read(*,*,IOSTAT=Reason) extra_epochs
                mul10=mod(extra_epochs,10)
      if(Reason.eq.0.and.mul10.eq.0.and.extra_epochs.gt.0) then
                valid=1
                else
      write(*,*)"======================================================"
      write(*,*)"Error: Ingrese una cantidad de ‚pocas num‚ricas"
      write(*,*)"enteras mayores a 0 y m£ltiplos de 10."
      write(*,*)"======================================================"
      end if
      end do
      write(*,*)""
      write(*,*) "Iniciando el entrenamiento... *beep beep boop*"
      call train(nn, x, d, n_puntos, extra_epochs)
      write(*,*) "Red entrenada!"
      write(*,*) "P‚rdida final:", loss(nn, x, d, n_puntos)
      write(*,*)""
      write(*,*)"======================================================"
      write(*,*)"Para conocer los resultados graficamente utilice un"
      write(*,*)"programa de confianza como Excel, Gnuplot, etc."
      write(*,*)"Este programa genera un archivo.csv llamado:"
      write(*,*)"'predictions.csv'"
      write(*,*)"======================================================"
      write(*,*)""
      write(*,*)"======================================================"
      write(*,*)"Ingrese 1 para entrenar m s su red neuronal"
      write(*,*)"o 0 para cerrar el programa."
      write(*,*)"======================================================"
            valid=0
            do while (valid.eq.0)
            read(*,*,IOSTAT=Reason) r
            if (reason.eq.0.and.(r.eq.0.or.r.eq.1)) then
              valid=1
              else
      write(*,*)"======================================================"
      write(*,*)"        Error: Ingrese una respuesta v lida."
      write(*,*)"======================================================"
            end if
            end do
	end do

	!Formato para escribir a archivo csv


	open (unit=2, file="predictions.csv", status="unknown")
	!write(*,*) "Ingrese el nÂ£mero de vectores a agregar."
	!read (*,*) n
	do i=1,n_puntos
		!write(*,*) "Ingrese el vector, x, y"
		a = FeedForward(nn, x(i))
		write(2, 100) x(i), a
100   format(f20.6,",",f20.6)
	end do

	close(2)

	end program

	function dotProduct(x, y, n)
		integer::n, i
		real::x,y, r
		dimension::x(n), y(n)
		r = 0.0
		i = 1
		do while(i<=n)
		    r = r + (x(i)*y(i))
		    i = i + 1
		end do
		        
		dotProduct = r
		return
	end function

	subroutine scalarMult(a, x, c, n)
		integer, intent(in)::n
		real, intent(in):: x(n)
	    real, intent(in)::a
	    real, intent(out)::c(n)
	    integer::i


	    i = 1
	    do while (i.le.n)
	      c(i) = x(i)*a

	      i = i + 1
	    end do 
		
	end subroutine

	subroutine linComb(a, x, b, y, c, n)
		integer, intent(in)::n
		real, intent(out):: c(n)
	    real, intent(in)::x(n), y(n)
	    real, intent(in)::a, b
	    integer::i

	    i = 1
	    do while (i.le.n)
	      c(i) = a*x(i)+ b*y(i)
	      i = i + 1
	    end do 
		
	end subroutine

	subroutine multArrays(x, y, c, n)
		integer, intent(in)::n
		real, intent(out):: c(n)
	    real, intent(in)::x(n), y(n)
	    integer::i

	    i = 1
	    do while (i<=n)
	      c(i) = x(i)*y(i)
	      i = i + 1
	    end do 
			
	end subroutine

	subroutine actFunction(x, c, n)
		integer, intent(in)::n
		real, intent(in):: x(n)
		real, intent(out)::c(n)
	    integer::i

	    i = 1
	    do while (i<=n)
	    ! c(i) = tanh(x(i))
	      c(i) = tanh(x(i))
	      i = i + 1
	    end do 
			
	end subroutine

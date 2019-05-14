	module class_nerualNetwork
	 	implicit none
	 	private
	 	public::nerualNetwork, init , FeedForward, loss, BackPropagate, train
	    
	 	type nerualNetwork
	 		integer::n
	        real::eta
	        real::b_2
	 		real::w_1(24), b_1(24), w_2(24)
	 		real::v_1(24), y_1(24), v_2
	 		real::r=0
	 	end type nerualNetwork

	contains
		subroutine init(this)
			type(nerualNetwork), intent(inout)::this
			integer::i=1
	        call random_seed()		
			!this%n = 24 !Define the number of neurons in the nn
			!this%eta = 0.01 !Define the gradient step learning
			call random_number(this%w_1)
			call random_number(this%w_2)

			i=1
			do while (i <= this%n)
				this%b_1(i) = 1
	            i = i + 1
			end do

			!this%b_2 = 1
		end subroutine

		function FeedForward(this, x) result(r)
			type(nerualNetwork), intent(inout)::this
			real::dotProduct
			real, intent(in)::x
			real::c(this%n)
			real::r
			integer::i=1
			
			!Calculate this%v_1
			call scalarMult(x, this%w_1, c, this%n)
			call linComb(1.0, c, 1.0, this%b_1, this%v_1, this%n)

			!calculate y_1
			!call actFunction(this%v_1, this%y_1, this%n)
			i=1
			do while(i<=this%n)
				this%y_1(i) = tanh(this%v_1(i))
				i = i + 1
			end do

			!Calculate v_2
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
			!real::temp(this%n), temp2(this%n)!, temp3(this%n)
			real::delta_1
	        dimension::delta_1(this%n)
			integer::i
			!write(*,*) this%w_1
			delta_out = d - y

			this%b_2 = this%b_2 + this%eta*delta_out

			i = 1
			do while(i<=this%n)
				!t = 1 - (tanh((this%eta*delta_out)*(this%y_1) + this%w_2(i)))**2
				this%w_2(i) = this%w_2(i) + (this%eta)*delta_out*(this%y_1(i))
				t = 1 - (tanh(this%v_1(i)))**2
				delta_1(i) = t*this%w_2(i)*delta_out
				i = i + 1
			end do

			!call scalarMult(this%eta*delta_out, this%y_1, temp, this%n)
			!call linComb(1.0, temp, 1.0, this%w_2, this%w_2, this%n)
			

			!do while(i<=this%n)
			!	temp(i) = tanh(this%v_1(i))
			!	i = i + 1
			!end do

			!call actFunction(this%v_1, temp, this%n)

			!i = 1
			!do while (i<=this%n)
			!	t = 1 - (temp(i))**2
			!	temp(i) = t
			!	i = i + 1
			!end do 

			!call multArrays(temp, this%w_2, temp2, this%n)

			!i = 1
			!do while(i<=this%n)
			!	delta_1(i) = temp(i)*this%w_2(i)*delta_out
			!	i = i + 1
			!end do

			!call scalarMult(delta_out, temp2, delta_1, this%n)

			i = 1
			do while(i<=this%n)
				!this%w_2(i) = this%w_2(i) + (this%eta)*delta_out*(this%y_1(i))
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
			write(*,*) "I got here!"
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
					write(*,*) "Epoch:", iter, " with loss:", loss(this, x, d, 300),&
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
	character::r
	real::x(300), y(300), v(300), d(300), a
	integer::i, extra_epochs

	call random_number(x)
	call random_number(y)
	call random_number(v)

	!i = 1
	!write(*,*) x
	!write(*,*) v


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

	nn = nerualNetwork(24, 0.005, 1, 0, 0, 0, 0, 0, 0, 0)

	write(*,*) "Start the program"
	read(*,*)
	write(*,*) "Calling the init method"
	call init(nn)
	write(*,*) "Initial loss", loss(nn, x, d, 300)
	!write(*,*) "Network initialized with the following values:"
	!write(*,*) max(0,2)
	!write(*,*) max(0, -0.2)
	!write(*,*) "n:", nn%n
	!write(*,*) "eta:", nn%eta
	!write(*,*) "b_2:", nn%b_2
	!write(*,*) "w_1:", nn%w_1
	!write(*,*) "w_2:", nn%w_2
	!write(*,*) "b_1:", nn%b_1
	!read(*,*)

	write(*,*) "Starting the training *beep beep boop*"
	call train(nn, x, d, 300, 5000)

	write(*,*) "Network trained!"
	write(*,*) "Final Loss", loss(nn, x, d, 300)

	write(*,*) "Do you want to train more your neural nerualNetwork?"
	read(*,*) r

	do while (r.eq."y")
		write(*,*) "Enter the number of extra epochs to train the nn"
		read(*,*) extra_epochs

		write(*,*) "Starting the training *beep beep boop*"
		call train(nn, x, d, 300, extra_epochs)

		write(*,*) "Do you want to train more your neural nerualNetwork? [y/n]"
		read(*,*) r
	end do

	!Formato para escribir a archivo csv


	open (unit=1, file="predictions.csv", status="unknown")
	!write(*,*) "Ingrese el n£mero de vectores a agregar."
	!read (*,*) n

	do i=1,300
		!write(*,*) "Ingrese el vector, x, y"
		a = FeedForward(nn, x(i))
		write(1, 100) x(i), a
100   format(f6.3,",",f6.3)
	end do

	close(1)  

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
	      c(i) = tanh(x(i))
	      i = i + 1
	    end do 
			
	end subroutine
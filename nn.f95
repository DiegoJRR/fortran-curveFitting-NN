module class_nerualNetwork
 	implicit none
 	private
 	public::nerualNetwork, init , FeedForward, loss!, BackPropagate, train
    
 	type nerualNetwork
 		integer::n=24
        real::eta=0.01
        real::b_2=1
 		real::w_1(24), b_1(24), w_2(24)
 		real::v_1(24), y_1(24), v_2(24)
 		real::r=0
 	end type nerualNetwork

contains
	subroutine init(this)
		type(nerualNetwork), intent(inout)::this
		integer::i=1
		!this%n = 24 !Define the number of neurons in the nn
		!this%eta = 0.01 !Define the gradient step learning
		call random_number(this%w_1)
		call random_number(this%w_2)

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
		
		!Calculate this%v_1
		call scalarMult(x, this%w_1, c, this%n)
		call linComb(1.0, c, 1.0, this%b_1, this%v_1, this%n)

		!calculate y_1
		call actFunction(this%v_1, this%y_1, this%n)

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

        temp = 0
		do while(i.le.n)
			c(i) = d(i) - FeedForward(this, x(i))
			temp = temp + (c(i)*c(i))
			i = i + 1
		end do

		cost = temp/n
	end function

	subroutine BackPropagate(this, x, y, d)
		type(nerualNetwork), intent(inout)::this
        real, intent(in)::x, y, d
		real::delta_out
		real::temp(this%n), temp2(this%n)
		real::delta_1(this%n)
		integer::i=1

		delta_out = d - y
		call scalarMult(this%eta*delta_out, this%y_1, temp, this%n)
		call linComb(1.0, temp, 1.0, this%w_2, this%w_2, this%n)
		this%b_2 = this%b_2 + this%eta*delta_out

		temp = 0
		call actFunction(this%v_1, temp, this%n)

		do while (i<=this%n)
			temp(i) = 1 - temp(i)*temp(i)
			i = i + 1
		end do 

		call multArrays(temp, this%w_2, temp2, this%n)
		call scalarMult(delta_out, temp2, delta_1)

		i = 1
		do while(i<=this%n)
			this%w_1(i) = this%w_1(i) + this%eta*x*delta_1(i)
			this%b_1(i) = this%b_1(i) + this%eta*delta_1(i)
			i = i + 1
		end do 
	end subroutine

end module class_nerualNetwork

program neural
use class_nerualNetwork
implicit none
type(nerualNetwork)::nn
real::x(300), y(300), v(300), d(300)
integer::i=1

call random_number(x)
call random_number(y)
call random_number(v)

do while (i<=300)
	d(i) = sin(20*x(i)) + 3*x(i) + v(i)
    i = i + 1
end do

nn = nerualNetwork(24, 0.01, 1, 0, 0, 0, 0, 0, 0, 0)
write(*,*) "Initial loss", loss(nn, x, d, 300)
write(*,*) nn%n
write(*,*) "Start the program"
read(*,*) 
write(*,*) "Calling the init method"
call init(nn)
write(*,*) "Network initialized with the following values:"
write(*,*) "n:", nn%n
write(*,*) "eta:", nn%eta
write(*,*) "b_2:", nn%b_2
write(*,*) "w_1:", nn%w_1
write(*,*) "w_2:", nn%w_2
write(*,*) "b_1:", nn%b_1
read(*,*)

end program

function dotProduct(x, y, n)
	integer::n, i=1
	real::x,y, r = 0.0
	dimension::x(n), y(n)

	do while(i.le.n)
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
    
    integer::i=1

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
    integer::i=1

    do while (i.le.n)
      c(i) = a*x(i)+ b*y(i)
      i = i + 1
    end do 
	
end subroutine

subroutine multArrays(x, y, c, n)
	integer, intent(in)::n
	real, intent(out):: c(n)
    real, intent(in)::x(n), y(n)
    integer::i=1

    do while (i.le.n)
      c(i) = x(i)*y(i)
      i = i + 1
    end do 
		
end subroutine

subroutine actFunction(x, c, n)
	integer, intent(in)::n
	real, intent(in):: x(n)
	real, intent(out)::c(n)
    integer::i=1

    do while (i.le.n)
      c(i) = tanh(x(i))
      i = i + 1
    end do 
		
end subroutine


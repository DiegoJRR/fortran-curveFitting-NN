module class_nerualNetwork
 	implicit none
 	private
 	public::nerualNetwork, init !, FeedForward, loss, BackPropagate, train
    
 	type nerualNetwork
 		integer::n
        real::eta
        real::b_2
 		real::w_1(24), b_1(24), w_2(24)
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
end module class_nerualNetwork

program neural
use class_nerualNetwork
implicit none
type(nerualNetwork)::nn
nn = nerualNetwork(24, 0.01, 1, 0, 0, 0)
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
    
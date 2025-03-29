class Autodiff{
public:
  Autodiff(std::shared_ptr<Network> n){
    network = n; 
  }    
private:
  std::shared_ptr<Network> network; 
}

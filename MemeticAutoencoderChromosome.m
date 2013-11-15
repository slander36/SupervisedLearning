classdef MemeticAutoencoderChromosome < handle
    % Defines the weight matrix used in a sparse autoencoder
    
    properties
        fitness = 0;
        
        autoEncoder % Holds weights for this autoencoder
        modified = true;   % Determines if the fitness need to be updated
        hiddenLayerSize % Hidden layer size
    end
    
    methods
        
        function chromo = MemeticAutoencoderChromosome(params, chromoToClone)
            if nargin < 1
                return;
            elseif nargin < 2
                % Initializes chromosome randomly
                chromo.autoEncoder = randn(params.hiddenLayerSize, 1);
                chromo.hiddenLayerSize = params.hiddenLayerSize;
            else
                % Make a deep copy of the given chromosome
                chromo.autoEncoder = chromoToClone.autoEncoder;
                chromo.hiddenLayerSize = chromoToClone.hiddenLayerSize;
            end
        end
   
        function mutate(self, pMutation)
            % Mutates this chromosome's tree by recursively traversing it
            if pMutation > rand()
                self.autoEncoder = self.autoEncoder .*...
                                   exp(randn(self.hiddenLayerSize, 1));
                self.modified = true;
            end
        end
        
        function offspring = crossover(parent1, parent2)
            % Performs crossover between two given parents
            offspring = MemeticAutoencoderChromosome([], parent1)
            blend = rand(self.hiddenLayerSize, 1);
            offspring.autoEncoder = offspring.autoEncoder .* blend +...
                                    parent2.autoEncoder .*...
                                    (repmat(1, parent1.hiddenLayerSize, 1) - blend);
            offspring.modified = true;
        end
        
    end
end


classdef MemeticAutoencoderChromosome < handle
    % Defines the weight matrix used in a sparse autoencoder
    
    properties
        fitness = 0;
        
        autoEncoder % Holds weights for this autoencoder
        modified = true;   % Determines if the fitness need to be updated
    end
    
    methods
        
        function chromo = MemeticAutoencoderChromosome(params, chromoToClone)
            if nargin < 1
                return;
            elseif nargin < 2
                % Initializes chromosome randomly
                chromo.autoEncoder = randn(params.hiddenLayerSize, 1);
            else
                % Make a deep copy of the given chromosome
                chromo.autoEncoder = chromoToClone.autoEncoder;
            end
        end
   
        function mutate(self, pMutation)
            % Mutates this chromosome's tree by recursively traversing it
            % TODO
        end
        
        function offspring = crossover(parent1, parent2)
            % Performs crossover between two given parents
            % TODO
        end
        
    end
end


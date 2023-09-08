LoadPackage("images", false);

lexMin := function(set, n)
    local Sym, elems, lsPerm, gen;

    lsPerm := [];
    Sym := SymmetricGroup(n);
    elems := Cartesian([1..n], [1..n]);
    for gen in GeneratorsOfGroup(Sym) do
        Add(lsPerm, PermListList( elems, List( elems, i -> OnPairs(i, gen) ) ));
    od;

    return MinimalImage(GroupByGenerators(lsPerm), set, OnSets);
end;
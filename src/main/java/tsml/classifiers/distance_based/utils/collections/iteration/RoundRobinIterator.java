package tsml.classifiers.distance_based.utils.collections.iteration;

import java.util.List;

/**
 * Purpose: round robin traverse a list.
 *
 * Contributors: goastler
 *
 * @param <A>
 */
public class RoundRobinIterator<A> extends LinearIterator<A> {
    public RoundRobinIterator(List<A> list) {
        super(list);
    }

    public RoundRobinIterator() {}

    @Override
    public A next() {
        A next = super.next();
        if(getIndex() == getList().size()) {
            setIndex(0);
        }
        return next;
    }

    @Override
    public void remove() {
        super.remove();
        if(getIndex() < 0) {
            setIndex(getList().size() - 1);
        }
    }

    protected boolean findHasNext() {
        return !getList().isEmpty();
    }

    @Override
    protected int findNextIndex() {
        return super.findNextIndex() % getList().size();
    }
}

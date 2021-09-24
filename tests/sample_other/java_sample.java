import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.lang.IllegalArgumentException;

class Dice {
    private final int count;
    private final int sides;
    private final int modifier;
    private final Random rng = new Random();

    public Dice(int count, int sides, int modifier) {
        if (count <= 0)
            throw new IllegalArgumentException(
                "Dice count must be greater than 0");
        if (sides <= 0)
            throw new IllegalArgumentException(
                "Dice sides must be greater than 0");

        this.count = count;
        this.sides = sides;
        this.modifier = modifier;

        // some nonsense to test tokenization:
        this.modifier();
        Dice.modifier;
        Dice.modifier();
    }
    public Dice(int count, int sides) {
        this(count, sides, 0);
    }

    public int roll() {
        int result = 0 + modifier;
        for (int i = 0; i < count; i++)
            result += rng.nextInt(sides) + 1;
        return result;
    }

    public double average() {
        return count*(sides+1)/2.0 + modifier;
    }
    public double average(boolean crit) {
        if (crit)
            return 2*count*(sides+1)/2.0 + modifier;
        else
            return average();
    }

    public int getDiceCount() { return count; }
    public int getSides() { return sides; }
    public int getModifier() { return modifier; }

    public static Dice fromString(String input) {
        String pattern = "(\\d*)\\s*d\\s*(\\d+)\\s*(([+-])\\s*(\\d+)|)";
        Pattern r = Pattern.compile(pattern);
        Matcher parsed = r.matcher(input);

        if (parsed.find()) {
            int count, sides, modifier;

            if (parsed.group(1).length() > 0)
                count = Integer.parseInt(parsed.group(1));
            else
                count = 1;
            sides = Integer.parseInt(parsed.group(2));

            if (parsed.group(3).length() == 0)
                modifier = 0;
            else {
                if (parsed.group(4).equals("+"))
                    modifier = Integer.parseInt(parsed.group(5));
                else
                    modifier = - Integer.parseInt(parsed.group(5));
            }
            return new Dice(count, sides, modifier);
        }
        else {
            throw new IllegalArgumentException(
                "Invalid dice specification string");
        }
    }

    @Override
    public String toString() {
        if (modifier >= 0)
            return count + "d" + sides + " + " + modifier;
        else
            return count + "d" + sides + " - " + modifier * -1;
    }
}

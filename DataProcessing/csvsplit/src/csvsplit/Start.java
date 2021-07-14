package csvsplit;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Start {

	// change this if the input data includes email or not
	public static String inputFileName = "set1q2";
	//change path to where your files are located
	public static String path = "yourrelativepath\\";
	//
	//

	public static boolean hasEmail;
	public static String titleString = "";
	public static List<String> siteNames = new ArrayList();
	public static List<String> questionTitles = new ArrayList<>();

	public static void main(String[] args) {
		prepareList();
		readCSV();
	}

	private static void prepareList() {
		if (inputFileName.equals("set2q1"))
			hasEmail = false;
		else
			hasEmail = true;
		siteNames.add("Flight search");
		siteNames.add("destination route search");
		siteNames.add("shopping");
		siteNames.add("news");

		questionTitles.add("id,");
		questionTitles.add("How familiar are you with this website? (low to high),");
		questionTitles.add("I think that I would like to use the websites on this set frequently (disagree to agree),");
		questionTitles.add("I found the websites in this set not unnecessarily complex. (hard to easy),");
		questionTitles.add("I thought  the websites in this set  were easy to use. (disagree to agree),");
		questionTitles.add(
				"I think that I would not need the support of an expert to be able to use  the websites in this set (disagree to agree),");
		questionTitles.add(
				"I found the various functions on  the websites in this set  were well integrated (disagree to agree),");
		questionTitles.add(
				"I thought there was not too much inconsistency on  the websites in this set (disagree to agree),");
		questionTitles.add(
				"I would imagine that most people would learn to use  the websites in this set  very quickly (disagree to agree),");
		questionTitles.add("I found  the websites in this set not very cumbersome to use (disagree to agree),");
		questionTitles.add("I felt very confident using  the websites in this set  (disagree to agree),");
		questionTitles.add(
				"I did not need to learn a lot of things before I could get going with the websites in this set (disagree to agree),");
		questionTitles.add("problems,");
		questionTitles.add("other");

		questionTitles.forEach(e -> {
			titleString += "" + e;
		});
		titleString += "\n";
	}

	private static void readCSV() {
		List<List<String>> rawInput = new ArrayList<>();
		try (BufferedReader br = new BufferedReader(
				new FileReader(path + inputFileName + ".csv"))) {
			String line;
			while ((line = br.readLine()) != null) {
				String[] values = line.split(",");
				rawInput.add(Arrays.asList(values));
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		List<List<String>> cleanedInput = new ArrayList<>();
		rawInput.remove(0);
		rawInput.forEach(e -> {
			if (e.size() > 10) {
				cleanedInput.add(e);				
			}
		});
		writeCSV(cleanedInput);
	}

	private static void writeCSV(List<List<String>> dataByUser) {
		int amountOfSpecificEntries = 44;
		// if there is an email the head contains 3 entries instead of 2
		int amountOfStaticEntries = 1;
		if (hasEmail)
			amountOfStaticEntries = 2;
		// output string
		String printstr = "";

		// iterate over all 4 websites
		for (int websiteIndex = 0; websiteIndex < 4; websiteIndex++) {
			// iterate over every participant
			printstr = titleString;
			for (int userIndex = 0; userIndex < dataByUser.size(); userIndex++) {
				// add title line to csv
				// iterate over every entry for this participant
				int valueIndex = 0;
				for (int entryIndex = 0; entryIndex < dataByUser.get(userIndex).size(); entryIndex++) {
					// filter out entries for this specific website (including email + id...) and
					// ignore the entries of the other websites
					// k <= 2 means time stamp, email, id
					// (k-3) % 4 == j means everything that is from website k
					// starting from k = 3 (k > 2) and always adding the last 2 since they apply to
					// all websites

					if (entryIndex == amountOfStaticEntries
							|| ((entryIndex - (amountOfStaticEntries + 1)) % 4 == websiteIndex
									&& entryIndex < (amountOfSpecificEntries + amountOfStaticEntries + 1)
									&& entryIndex >= amountOfStaticEntries + 1)
							|| entryIndex >= (amountOfSpecificEntries + amountOfStaticEntries + 1)) {
						valueIndex++;
						String entry = dataByUser.get(userIndex).get(entryIndex);
						if (valueIndex == 3 || valueIndex == 6 || valueIndex == 8 || valueIndex == 10
								|| valueIndex == 12) {
							entry = cleanUpValue(entry, valueIndex);
						}
						if (entryIndex != dataByUser.get(userIndex).size() - 1)
							printstr += "" + entry + ",";
						else
							// if it is the last entry, don't add comma
							printstr += "" + entry;
					}
				}
				if (userIndex < dataByUser.size() - 1)
					printstr += "\n";
			}
			// write file with long filename
			try (PrintWriter out = new PrintWriter(path+"Desktop\\output\\" + inputFileName + "\\site"
					+ (websiteIndex + 1) + "_" + siteNames.get(websiteIndex) + ".csv")) {
				out.println(printstr);
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}
	}

	public static String cleanUpValue(String rawValue, int counter) {
		counter--;
		if (counter == 2)
			return (mapValue(rawValue.replaceAll("^\"+|\"+$", "")));
		if (counter == 3 || counter == 5 || counter == 7 || counter == 9)
			return reverseValue(rawValue.replaceAll("^\"+|\"+$", ""));
		if (counter == 11)
			return reverseValue(mapValue(rawValue.replaceAll("^\"+|\"+$", "")));
		return "error";
	}

	public static String reverseValue(String rV) {
		return Integer.toString((8 - Integer.parseInt(rV)));
	}

	public static String mapValue(String rV) {
		switch (Integer.parseInt(rV.replaceAll("^\"+|\"+$", ""))) {
		case 1:
			return "1";
		case 2:
			return "2";
		case 3:
			return "4";
		case 4:
			return "6";
		case 5:
			return "7";
		}
		return "error";
	}

}
